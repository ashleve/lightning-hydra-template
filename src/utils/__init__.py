import logging
import os
import warnings
from typing import Any, Dict, List, Sequence

import dotenv
import hydra
import pytorch_lightning as pl
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only


def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


log = get_logger(__name__)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities, controlled by config flags.

    Utilities:
    - Loading environment variables from .env file
    - Ignoring python warnings
    - Converting relative ckpt path to absolute path
    - Setting global seeds
    - Rich config printing
    """

    # load environment variables from `.env` file if exists
    if cfg.get("load_dotenv"):
        log.info(f"Loading environment variables! <cfg.load_dotenv={cfg.load_dotenv}>")
        dotenv.load_dotenv(override=True, verbose=True)

    # disable python warnings
    if cfg.get("ignore_warnings"):
        log.info(f"Disabling python warnings! <cfg.ignore_warnings={cfg.ignore_warnings}>")
        warnings.filterwarnings("ignore")

    # convert relative ckpt path to absolute path
    # otherwise relative paths won't work since hydra hijacks the work dir
    if cfg.get("ckpt_path") and not os.path.isabs(cfg.ckpt_path):
        log.info(f"Converting ckpt path to absolute path! <cfg.ckpt_path={cfg.ckpt_path}>")
        cfg.ckpt_path = os.path.join(hydra.utils.get_original_cwd(), cfg.ckpt_path)

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        log.info(f"Setting seeds! <cfg.seed={cfg.seed}>")
        pl.seed_everything(cfg.seed, workers=True)

    # pretty print config tree using Rich library
    if cfg.get("print_config"):
        log.info(f"Printing config tree with Rich! <cfg.print_config={cfg.print_config}>")
        print_config(cfg, resolve=True)


@rank_zero_only
def print_config(
    cfg: DictConfig,
    print_order: Sequence[str] = ("datamodule", "model", "callbacks", "logger", "trainer"),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
        print_order (Sequence[str], optional): Determines in what order config components are printed.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    for field in print_order:
        queue.append(field) if field in cfg else log.info(
            f"Field '{field}' not found in config. Skipping '{field}' config printing..."
        )

    for field in cfg:
        if field not in queue:
            queue.append(field)

    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.log", "w") as file:
        rich.print(tree, file=file)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config."""
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        return callbacks

    assert isinstance(callbacks_cfg, DictConfig), "Callbacks config must be a DictConfig!"

    for _, cb_conf in callbacks_cfg.items():
        if "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[LightningLoggerBase]:
    """Instantiates loggers from config."""
    logger: List[LightningLoggerBase] = []

    if not logger_cfg:
        return logger

    assert isinstance(logger_cfg, DictConfig), "Loggers config must be a DictConfig!"

    for _, lg_conf in logger_cfg.items():
        if "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


@rank_zero_only
def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionaly saves:
    - number of model parameters
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

    if "seed" in cfg:
        hparams["seed"] = cfg["seed"]
    if "callbacks" in cfg:
        hparams["callbacks"] = cfg["callbacks"]
    if "ckpt_path" in cfg:
        hparams["ckpt_path"] = cfg.ckpt_path

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)


def get_metric_value(metric_name: str, trainer: Trainer) -> float:
    """Retrieves value of the metric logged in LightningModule.

    Args:
        metric_name (str): Name of the metric.
        trainer (Trainer): Lightning Trainer instance.
    """
    if metric_name not in trainer.callback_metrics:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )
    return trainer.callback_metrics[metric_name]


def finish(object_dict: Dict[str, Any]) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash
    logger = object_dict.get("logger", [])
    for lg in logger:
        if isinstance(lg, pl.loggers.wandb.WandbLogger):
            import wandb

            wandb.finish()
