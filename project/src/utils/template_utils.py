# pytorch lightning imports
import wandb
from pytorch_lightning.loggers.wandb import WandbLogger
import pytorch_lightning as pl

# hydra imports
from omegaconf import DictConfig, OmegaConf

# normal imports
from typing import List
import logging

log = logging.getLogger(__name__)


def print_config(config: DictConfig):
    log.info(f"\n{OmegaConf.to_yaml(config, resolve=True)}")


def print_module_init_info(model, datamodule, callbacks, loggers, trainer):
    message = "Model initialised:" + "\n" + model.__module__ + "." + model.__class__.__name__ + "\n"
    log.info(message)

    message = "Datamodule initialised:" + "\n" + datamodule.__module__ + "." + datamodule.__class__.__name__ + "\n"
    log.info(message)

    message = "Callbacks initialised:" + "\n"
    for cb in callbacks:
        message += cb.__module__ + "." + cb.__class__.__name__ + "\n"
    log.info(message)

    message = "Loggers initialised:" + "\n"
    for logger in loggers:
        message += logger.__module__ + "." + logger.__class__.__name__ + "\n"
    log.info(message)

    message = "Trainer initialised:" + "\n" + trainer.__module__ + "." + trainer.__class__.__name__ + "\n"
    log.info(message)


def make_wandb_watch_model(loggers: List[pl.loggers.LightningLoggerBase], model: pl.LightningModule):
    for logger in loggers:
        if isinstance(logger, WandbLogger):
            if hasattr(model, 'architecture'):
                logger.watch(model.architecture, log=None)
            else:
                logger.watch(model, log=None)


def send_hparams_to_loggers(loggers: List[pl.loggers.LightningLoggerBase], hparams: dict):
    for logger in loggers:
        logger.log_hyperparams(hparams)


def log_hparams(config, model, datamodule, callbacks, loggers, trainer):
    hparams = {
        "_class_model": config["model"]["_target_"],
        "_class_datamodule": config["datamodule"]["_target_"]
    }

    if hasattr(model, "architecture"):
        obj = model.architecture
        hparams["_class_model_architecture"] = obj.__module__ + "." + obj.__class__.__name__

    hparams.update(config["seeds"])
    hparams.update(config["model"])
    hparams.update(config["datamodule"])
    hparams.update(config["trainer"])
    hparams.pop("_target_")

    if hasattr(datamodule, 'data_train') and datamodule.data_train is not None:
        hparams["train_size"] = len(datamodule.data_train)
    if hasattr(datamodule, 'data_val') and datamodule.data_val is not None:
        hparams["val_size"] = len(datamodule.data_val)
    if hasattr(datamodule, 'data_test') and datamodule.data_test is not None:
        hparams["test_size"] = len(datamodule.data_test)

    send_hparams_to_loggers(loggers=loggers, hparams=hparams)


def extras(config, model, datamodule, callbacks, loggers, trainer):
    # Print info about which modules were initialized
    print_module_init_info(model, datamodule, callbacks, loggers, trainer)

    # Log extra hyperparameters to loggers
    log_hparams(config, model, datamodule, callbacks, loggers, trainer)

    # If WandbLogger was initialized, make it watch the model
    make_wandb_watch_model(loggers=loggers, model=model)


def finish():
    wandb.finish()
