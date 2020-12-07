from pytorch_lightning.loggers import WandbLogger
from argparse import ArgumentParser
import pytorch_lightning as pl
from typing import List
import yaml
import os

# utils
from utils.init_utils import init_lit_model, init_data_module, init_trainer, init_callbacks, init_wandb_logger


def train(project_config: dict, run_config: dict, use_wandb: bool):
    # Init PyTorch Lightning model ⚡
    lit_model: pl.LightningModule = init_lit_model(hparams=run_config["model"])

    # Init PyTorch Lightning datamodule ⚡
    datamodule: pl.LightningDataModule = init_data_module(hparams=run_config["dataset"])

    # Init Weights&Biases logger
    logger: pl.loggers.WandbLogger = init_wandb_logger(
        project_config=project_config,
        run_config=run_config,
        lit_model=lit_model,
        datamodule=datamodule,
        log_path=os.path.join(os.path.dirname(__file__), "logs/")
    ) if use_wandb else None

    # Init callbacks
    callbacks: List[pl.Callback] = init_callbacks(
        project_config=project_config,
        run_config=run_config,
        use_wandb=use_wandb
    )

    # Init PyTorch Lightning trainer ⚡
    trainer: pl.Trainer = init_trainer(
        project_config=project_config,
        run_config=run_config,
        logger=logger,
        callbacks=callbacks
    )

    # Evaluate model on test set before training
    # trainer.test(model=lit_model, datamodule=datamodule)

    # Train the model
    trainer.fit(model=lit_model, datamodule=datamodule)

    # Evaluate model on test set after training
    trainer.test()


def load_config(filename):
    path = os.path.join(os.path.dirname(__file__), filename)
    with open(path, "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return config


def main(run_config_name: str, use_wandb: bool):
    # Load configs
    project_config: dict = load_config("project_config.yaml")
    run_config: dict = load_config("run_configs.yaml")[run_config_name]

    # Train model
    train(project_config=project_config, run_config=run_config, use_wandb=use_wandb)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-r", "--run_config", type=str, default="MNIST_CLASSIFIER_V2")
    parser.add_argument("-n", "--no_wandb", dest='use_wandb', action='store_false')
    parser.set_defaults(use_wandb=True)
    args = parser.parse_args()

    main(run_config_name=args.run_config, use_wandb=args.use_wandb)
