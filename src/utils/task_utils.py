import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only

from src.utils import pylogger, rich_utils

log = pylogger.get_pylogger(__name__)


def task_wrapper(task_func) -> Callable:
    """Optional decorator that wraps the task function in extra utilities.

    Utilities:
    - Calling the task_utils.start() before the task is started
    - Calling the task_utils.finish() after the task is finished
    - Logging the total time of execution
    - Enabling repeating task execution on failure
    """

    def wrap(cfg: DictConfig):
        start_time = time.time()

        # apply optional config utilities
        start(cfg)

        # TODO: repeat call if fails...
        result = task_func(cfg=cfg)
        metric_value, object_dict = result

        # make sure everything closed properly
        finish(object_dict)

        # save task execution time
        end_time = time.time()
        save_exec_time(cfg.paths.output_dir, cfg.task_name, end_time - start_time)

        # make sure returned types are correct
        if not (isinstance(metric_value, float) or metric_value is None):
            raise TypeError("Incorrect type of 'metric_value'.")
        if not isinstance(object_dict, dict):
            raise TypeError("Incorrect type of 'object_dict'.")

        return metric_value, object_dict

    return wrap


def start(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Setting global seeds
    - Rich config printing
    """

    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.extras.get("seed"):
        log.info(f"Setting seeds! <cfg.extras.seed={cfg.extras.seed}>")
        pl.seed_everything(cfg.extras.seed, workers=True)


def finish(object_dict: Dict[str, Any]) -> None:
    """Applies optional utilities after the task is executed.

    Utilities:
    - Making sure all loggers closed properly (prevents logging failure during multirun)
    """
    for logger in object_dict.get("logger", []):

        if isinstance(logger, pl.loggers.wandb.WandbLogger):
            import wandb

            wandb.finish()

        if isinstance(logger, pl.loggers.neptune.NeptuneLogger):
            import neptune

            neptune.stop()

        if isinstance(logger, pl.loggers.mlflow.MLFlowLogger):
            import mlflow

            mlflow.end_run()

        if isinstance(logger, pl.loggers.comet.CometLogger):
            logger._experiment.end()


@rank_zero_only
def save_exec_time(path, task_name, time_in_seconds) -> None:
    """Saves task execution time to file."""
    with open(Path(path, "exec_time.log"), "w+") as file:
        file.write("Total task execution time.\n")
        file.write(task_name + ": " + str(time_in_seconds) + " (s)" + "\n")


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config."""
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("Callbacks config is empty.")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[LightningLoggerBase]:
    """Instantiates loggers from config."""
    logger: List[LightningLoggerBase] = []

    if not logger_cfg:
        log.warning("Logger config is empty.")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Loggers config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


@rank_zero_only
def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """Controls which config parts are saved by lightning loggers.

    Additionaly saves:
    - Number of model parameters
    """

    hparams = {}

    cfg = object_dict["cfg"]
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["datamodule"] = cfg["datamodule"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["seed"] = cfg.get("seed")
    hparams["task_name"] = cfg.get("task_name")
    hparams["exp_name"] = cfg.get("name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)


def get_metric_value(metric_name: str, trainer: Trainer) -> float:
    """Retrieves value of the metric logged in LightningModule."""
    if metric_name not in trainer.callback_metrics:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )
    return trainer.callback_metrics[metric_name].item()
