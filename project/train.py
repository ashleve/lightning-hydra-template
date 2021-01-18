# pytorch lightning imports
from pytorch_lightning.loggers import LightningLoggerBase
import pytorch_lightning as pl
import torch

# hydra imports
from omegaconf import DictConfig
import hydra

# normal imports
from typing import List
import wandb

# template utils imports
import src.utils.initializers as utils


def train(config):
    # Set global PyTorch seed
    if "seeds" in config and "pytorch_seed" in config["seeds"]:
        torch.manual_seed(seed=config["seeds"]["pytorch_seed"])

    # Init PyTorch Lightning model ⚡
    model: pl.LightningModule = utils.init_model(model_config=config["model"])

    # Init PyTorch Lightning datamodule ⚡
    datamodule: pl.LightningDataModule = utils.init_datamodule(datamodule_config=config["datamodule"])

    # Init PyTorch Lightning callbacks ⚡
    callbacks = []
    if "callbacks" in config:
        callbacks: List[pl.Callback] = utils.init_callbacks(callbacks_config=config["callbacks"])

    # Init PyTorch Lightning loggers ⚡
    loggers = []
    if "logger" in config:
        loggers: List[pl.loggers.LightningLoggerBase] = utils.init_loggers(loggers_config=config["logger"])

    # Init PyTorch Lightning trainer ⚡
    trainer: pl.Trainer = utils.init_trainer(trainer_config=config["trainer"], callbacks=callbacks, loggers=loggers)

    # Print info in terminal about all succesfully initialized objects
    utils.print_module_init_info(model=model, datamodule=datamodule, callbacks=callbacks, loggers=loggers)

    # If WandbLogger was initialized, make it watch the model
    utils.make_wandb_watch_model(loggers=loggers, model=model)

    # Make loggers save hyperparameters specified in 'extra_logs' section of config
    utils.log_hparams(loggers=loggers, config=config, model=model, datamodule=datamodule, callbacks=callbacks)

    # Train the model
    trainer.fit(model=model, datamodule=datamodule)

    # Evaluate model on test set after training
    trainer.test()

    # Finish wandb run if WandbLogger was initialized (otherwise next run in hydra multirun would crash)
    wandb.finish()


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
    utils.print_config(config)
    train(config)


if __name__ == "__main__":
    main()
