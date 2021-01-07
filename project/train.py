# pytorch lightning imports
from pytorch_lightning.loggers import LightningLoggerBase
import pytorch_lightning as pl
import torch

# hydra imports
from omegaconf import DictConfig, OmegaConf
import hydra.conf.hydra.output
import hydra

# normal imports
from typing import List
import logging

# template utils imports
from template_utils.initializers import (
    validate_config,
    init_model,
    init_datamodule,
    init_callbacks,
    init_loggers,
    init_trainer
)

log = logging.getLogger(__name__)


def train(config):
    validate_config(config)

    # Set global PyTorch seed
    if "seeds" in config and "pytorch_seed" in config["seeds"]:
        torch.manual_seed(config["seeds"]["pytorch_seed"])

    # Init PyTorch Lightning model ⚡
    model: pl.LightningModule = init_model(
        model_config=config["model"],
        optimizer_config=config["optimizer"]
    )

    # Init PyTorch Lightning datamodule ⚡
    datamodule: pl.LightningDataModule = init_datamodule(
        datamodule_config=config["datamodule"],
        data_dir=config["data_dir"]
    )

    # Init PyTorch Lightning callbacks ⚡
    callbacks: List[pl.Callback] = init_callbacks(config=config)

    # Init PyTorch Lightning logger ⚡
    loggers: List[pl.loggers.LightningLoggerBase] = init_loggers(
        config=config,
        model=model,
        datamodule=datamodule
    )

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

    if "seeds" in config and "pytorch_seed" in config["seeds"]:
        log.info(f"Using pytorch seed: {config['seeds']['pytorch_seed']}\n")

    # Init PyTorch Lightning trainer ⚡
    trainer: pl.Trainer = init_trainer(
        trainer_config=config["trainer"],
        callbacks=callbacks,
        loggers=loggers
    )

    # Evaluate model on test set before training
    # trainer.test(model=model, datamodule=datamodule)

    # Train the model
    trainer.fit(model=model, datamodule=datamodule)

    # Evaluate model on test set after training
    trainer.test()


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig) -> None:
    log.info(f"\n{OmegaConf.to_yaml(config, resolve=True)}")  # print content of config
    train(config)


if __name__ == "__main__":
    main()
