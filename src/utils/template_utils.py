import logging
import warnings
from typing import List

import pytorch_lightning as pl
import wandb
from hydra.utils import get_original_cwd, log, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers.wandb import WandbLogger
from rich import print
from rich.syntax import Syntax
from rich.tree import Tree


def extras(config: DictConfig):
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
    pl_logger = logging.getLogger('lightning')
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


def print_config(config: DictConfig, resolve: bool = True):
    """Prints content of Hydra config using Rich library.

    Args:
        config (DictConfig): [description]
        resolve (bool, optional): Whether to resolve reference fields in Hydra config.
    """

    # TODO print main config path and experiment config path
    # directory = to_absolute_path("configs/config.yaml")
    # print(f"Main config path: [link file://{directory}]{directory}")
    
    #TODO make this method more general

    style = "dim"

    tree = Tree(f":gear: TRAINING CONFIG", style=style, guide_style=style)

    trainer = OmegaConf.to_yaml(config["trainer"], resolve=resolve)
    trainer_branch = tree.add("Trainer", style=style, guide_style=style)
    trainer_branch.add(Syntax(trainer, "yaml"))

    model = OmegaConf.to_yaml(config["model"], resolve=resolve)
    model_branch = tree.add("Model", style=style, guide_style=style)
    model_branch.add(Syntax(model, "yaml"))

    datamodule = OmegaConf.to_yaml(config["datamodule"], resolve=resolve)
    datamodule_branch = tree.add("Datamodule", style=style, guide_style=style)
    datamodule_branch.add(Syntax(datamodule, "yaml"))
    
    optimizer = OmegaConf.to_yaml(config["optimizer"], resolve=resolve)
    optimizer_branch = tree.add("Optimizer", style=style, guide_style=style)
    optimizer_branch.add(Syntax(optimizer, "yaml"))

    callbacks_branch = tree.add("Callbacks", style=style, guide_style=style)
    if "callbacks" in config:
        for cb_name, cb_conf in config["callbacks"].items():
            cb = callbacks_branch.add(cb_name, style=style, guide_style=style)
            cb.add(Syntax(OmegaConf.to_yaml(cb_conf, resolve=resolve), "yaml"))
    else:
        callbacks_branch.add("None")

    logger_branch = tree.add("Logger", style=style, guide_style=style)
    if "logger" in config:
        for lg_name, lg_conf in config["logger"].items():
            lg = logger_branch.add(lg_name, style=style, guide_style=style)
            lg.add(Syntax(OmegaConf.to_yaml(lg_conf, resolve=resolve), "yaml"))
    else:
        logger_branch.add("None")

    seed = config.get("seed", "None")
    seed_branch = tree.add(f"Seed", style=style, guide_style=style)
    seed_branch.add(str(seed) + "\n")

    print(tree)


def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
):
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
    # (this is just to prevent trainer logging hparams of model as we manage it ourselves)
    for lg in logger:
        lg.log_hyperparams = lambda x: None


def finish(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
):
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
