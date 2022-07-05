import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List

import hydra
import pkg_resources
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
    - Calling the `extras()` before the task is started
    - Calling the `close_loggers()` after the task is finished
    - Calling the `close_loggers()` if exception occurs (otherwise multirun with logger could fail)
    - Logging the exception if occurs
    - Logging the task total execution time
    """

    def wrap(cfg: DictConfig):

        # apply extra config utilities
        extras(cfg)

        start_time = time.time()

        # execute the task
        try:
            result = task_func(cfg=cfg)
        except Exception as ex:
            log.exception("")
            close_loggers()
            raise ex

        end_time = time.time()

        # save task execution time
        content = f"'{cfg.task_name}' execution time: {end_time - start_time} (s)"
        path = Path(cfg.paths.output_dir, "exec_time.log")
        save_file(path, content)

        # make sure loggers closed properly
        close_loggers()

        # make sure returned types are correct
        metric_value, object_dict = result
        if not (isinstance(metric_value, float) or metric_value is None):
            raise TypeError("Incorrect type of 'metric_value'.")
        if not isinstance(object_dict, dict):
            raise TypeError("Incorrect type of 'object_dict'.")

        return metric_value, object_dict

    return wrap


def extras(cfg: DictConfig) -> None:
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


@rank_zero_only
def save_file(path, content) -> None:
    """Save file in rank zero mode."""
    with open(path, "w+") as file:
        file.write(content)


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
        raise TypeError("Logger config must be a DictConfig!")

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


def close_loggers() -> None:
    """Makes sure all loggers closed properly (prevents logging failure during multirun)."""

    def package_available(package_name: str) -> bool:
        try:
            return pkg_resources.require(package_name) is not None
        except pkg_resources.DistributionNotFound:
            return False

    if package_available("wandb"):
        from wandb import finish

        finish()

    if package_available("neptune"):
        from neptune import stop

        stop()

    if package_available("mlflow"):
        from mlflow import end_run

        end_run()
