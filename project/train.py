# pytorch lightning imports
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning import LightningModule, LightningDataModule, Callback, Trainer
import torch

# hydra imports
from omegaconf import DictConfig
import hydra

# normal imports
from typing import List

# template utils imports
from src.utils import template_utils as utils


def train(config):
    # Set global PyTorch seed
    if "seeds" in config and "pytorch_seed" in config["seeds"]:
        torch.manual_seed(seed=config["seeds"]["pytorch_seed"])

    # Init PyTorch Lightning model ⚡
    model: LightningModule = hydra.utils.instantiate(config["model"])

    # Init PyTorch Lightning datamodule ⚡
    datamodule: LightningDataModule = hydra.utils.instantiate(config["datamodule"])
    datamodule.prepare_data()
    datamodule.setup()

    # Init PyTorch Lightning callbacks ⚡
    callbacks: List[Callback] = [
        hydra.utils.instantiate(callback_conf)
        for callback_name, callback_conf in config["callbacks"].items()
    ] if "callbacks" in config else []

    # Init PyTorch Lightning loggers ⚡
    loggers: List[LightningLoggerBase] = [
        hydra.utils.instantiate(logger_conf)
        for logger_name, logger_conf in config["logger"].items()
        if "_target_" in logger_conf   # ignore logger conf if there's no target
    ] if "logger" in config else []

    # Init PyTorch Lightning trainer ⚡
    trainer: Trainer = hydra.utils.instantiate(config["trainer"], callbacks=callbacks, logger=loggers)

    # Magic
    utils.extras(config, model, datamodule, callbacks, loggers, trainer)

    # Train the model
    trainer.fit(model=model, datamodule=datamodule)

    # Evaluate model on test set after training
    trainer.test()

    # Finish run
    utils.finish()


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
    utils.print_config(config)
    train(config)


if __name__ == "__main__":
    main()
