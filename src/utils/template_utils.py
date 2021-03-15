import logging
import warnings
from typing import List, Sequence

import pytorch_lightning as pl
import wandb
from hydra.utils import log
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers.wandb import WandbLogger
from rich import print
from rich.syntax import Syntax
from rich.tree import Tree


def extras(config: DictConfig) -> None:
    """A couple of optional utilities, controlled by main config file.
        - disabling warnings
        - disabling lightning logs
        - easier access to debug mode
        - forcing debug friendly configuration
    Args:
        config (DictConfig): [description]
    """

    # make it possible to add new keys to config
    OmegaConf.set_struct(config, False)

    # fix double logging bug (this will be removed when lightning releases patch)
    pl_logger = logging.getLogger("lightning")
    pl_logger.propagate = False

    # [OPTIONAL] Disable python warnings if <config.disable_warnings=True>
    if config.get("disable_warnings"):
        log.info(f"Disabling python warnings! <{config.disable_warnings=}>")
        warnings.filterwarnings("ignore")

    # [OPTIONAL] Disable Lightning logs if <config.disable_lightning_logs=True>
    if config.get("disable_lightning_logs"):
        log.info(f"Disabling lightning logs! {config.disable_lightning_logs=}>")
        logging.getLogger("lightning").setLevel(logging.ERROR)

    # [OPTIONAL] Set <config.trainer.fast_dev_run=True> if  <config.debug=True>
    if config.get("debug"):
        log.info(f"Running in debug mode! <{config.debug=}>")
        config.trainer.fast_dev_run = True

    # [OPTIONAL] Force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    if config.trainer.get("fast_dev_run"):
        log.info(
            f"Forcing debugger friendly configuration! "
            f"<{config.trainer.fast_dev_run=}>"
        )
        # Debuggers don't like GPUs or multiprocessing
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0

    # disable adding new keys to config
    OmegaConf.set_struct(config, True)


def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "trainer",
        "model",
        "optimizer",
        "datamodule",
        "callbacks",
        "logger",
        "seed",
    ),
    extra_depth_fields: Sequence[str] = (
        "callbacks",
        "logger",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Config.
        fields (Sequence[str], optional): Determines which main fields from config will be printed
        and in what order.
        extra_depth_fields (Sequence[str], optional): Fields which should be printed with extra tree depth.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    # TODO print main config path and experiment config path
    # print(f"Main config path: [link file://{directory}]{directory}")
    
    # TODO refactor the whole method
        
    style = "dim"

    tree = Tree(f":gear: CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)

        if not config_section:
            # raise Exception(f"Field {field} not found in config!")
            branch.add("None")
            continue

        if field in extra_depth_fields:
            for nested_field in config_section:
                nested_config_section = config_section[nested_field]
                nested_branch = branch.add(nested_field, style=style, guide_style=style)
                cfg_str = OmegaConf.to_yaml(nested_config_section, resolve=resolve)
                nested_branch.add(Syntax(cfg_str, "yaml"))
        else:
            if isinstance(config_section, DictConfig):
                cfg_str = OmegaConf.to_yaml(config_section, resolve=resolve)
            else:
                cfg_str = str(config_section)
            branch.add(Syntax(cfg_str, "yaml"))

    print(tree)


def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.

    Additionaly saves:
        - sizes of train, val, test dataset
        - number of trainable model parameters

    Args:
        config (DictConfig): [description]
        model (pl.LightningModule): [description]
        datamodule (pl.LightningDataModule): [description]
        trainer (pl.Trainer): [description]
        callbacks (List[pl.Callback]): [description]
        logger (List[pl.loggers.LightningLoggerBase]): [description]
    """

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["trainer"] = config["trainer"]
    hparams["model"] = config["model"]
    hparams["optimizer"] = config["optimizer"]
    hparams["datamodule"] = config["datamodule"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # save sizes of each dataset
    # (requires calling `datamodule.setup()` first to initialize datasets)
    # datamodule.setup()
    # if hasattr(datamodule, "data_train") and datamodule.data_train:
    #     hparams["datamodule/train_size"] = len(datamodule.data_train)
    # if hasattr(datamodule, "data_val") and datamodule.data_val:
    #     hparams["datamodule/val_size"] = len(datamodule.data_val)
    # if hasattr(datamodule, "data_test") and datamodule.data_test:
    #     hparams["datamodule/test_size"] = len(datamodule.data_test)

    # save number of model parameters
    hparams["model/params_total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

    # disable logging any more hyperparameters for all loggers
    # (this is just a trick to prevent trainer from logging hparams of model, since we already did that above)
    trainer.logger.log_hyperparams = lambda params: None


def finish(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """Makes sure everything closed properly.

    Args:
        config (DictConfig): [description]
        model (pl.LightningModule): [description]
        datamodule (pl.LightningDataModule): [description]
        trainer (pl.Trainer): [description]
        callbacks (List[pl.Callback]): [description]
        logger (List[pl.loggers.LightningLoggerBase]): [description]
    """

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, WandbLogger):
            wandb.finish()
