# pytorch lightning imports
from pytorch_lightning.loggers import LightningLoggerBase
import pytorch_lightning as pl
import torch

# hydra imports
from omegaconf import DictConfig, OmegaConf
import hydra.conf.hydra.output

# normal imports
from typing import List
import logging
import os

# template utils imports
from template_utils.initializers import (
    format_config_paths,
    init_model,
    init_datamodule,
    init_callbacks,
    init_loggers,
    init_trainer
)

BASE_DIR: str = os.path.dirname(__file__)
LOG = logging.getLogger(__name__)


def train(config):

    # Set global PyTorch seed
    if "seeds" in config and "pytorch_seed" in config["seeds"]:
        torch.manual_seed(config["seeds"]["pytorch_seed"])

    # Covert paths to absolute and normalize them
    config = format_config_paths(config=config, base_dir=BASE_DIR)

    # Init PyTorch Lightning model ⚡
    model: pl.LightningModule = init_model(
        model_config=config["model"],
        optimizer_config=config["optimizer"]
    )

    # Init PyTorch Lightning datamodule ⚡
    datamodule: pl.LightningDataModule = init_datamodule(
        datamodule_config=config["datamodule"],
        data_dir=config["paths"]["data_dir"]
    )

    # Init PyTorch Lightning callbacks ⚡
    callbacks: List[pl.Callback] = init_callbacks(config=config)

    # Init PyTorch Lightning loggers ⚡
    loggers: List[pl.loggers.LightningLoggerBase] = init_loggers(
        config=config,
        model=model,
        datamodule=datamodule
    )

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
    LOG.info(OmegaConf.to_yaml(config, resolve=True))  # print content of config
    train(config)


if __name__ == "__main__":
    main()
