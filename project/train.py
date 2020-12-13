# pytorch lightning imports
from pytorch_lightning import Trainer, LightningModule, LightningDataModule, Callback
from pytorch_lightning.loggers import LightningLoggerBase
import torch

# normal imports
from argparse import ArgumentParser
from typing import List
import pprint
import yaml
import os

# template utils imports
from template_utils.initializers import (
    normalize_config_paths,
    init_model,
    init_datamodule,
    init_callbacks,
    init_loggers,
    init_trainer
)


# Everything will be loaded relatively to placement of 'train.py' file!
BASE_DIR: str = os.path.dirname(__file__)


def train(project_config: dict, run_config: dict, use_wandb: bool):

    # Set global PyTorch seed
    if "seed" in run_config:
        torch.manual_seed(run_config["seed"])

    # Covert paths to absolute and normalize them
    project_config, run_config = normalize_config_paths(
        project_config=project_config,
        run_config=run_config,
        base_dir=BASE_DIR
    )

    # Init PyTorch Lightning model ⚡
    model: LightningModule = init_model(
        model_config=run_config["model"]
    )

    # Init PyTorch Lightning datamodule ⚡
    datamodule: LightningDataModule = init_datamodule(
        datamodule_config=run_config["datamodule"],
        data_dir=project_config["data_dir"]
    )

    # Init PyTorch Lightning callbacks ⚡
    callbacks: List[Callback] = init_callbacks(
        project_config=project_config,
        run_config=run_config,
        use_wandb=use_wandb,
        base_dir=BASE_DIR
    )

    # Init PyTorch Lightning loggers ⚡
    loggers: List[LightningLoggerBase] = init_loggers(
        project_config=project_config,
        run_config=run_config,
        model=model,
        datamodule=datamodule,
        use_wandb=use_wandb
    )

    # Init PyTorch Lightning trainer ⚡
    trainer: Trainer = init_trainer(
        project_config=project_config,
        run_config=run_config,
        callbacks=callbacks,
        loggers=loggers
    )

    # Evaluate model on test set before training
    # trainer.test(model=model, datamodule=datamodule)

    # Train the model
    trainer.fit(model=model, datamodule=datamodule)

    # Evaluate model on test set after training
    trainer.test()


def load_config(path):
    abs_path = os.path.join(BASE_DIR, path) if not os.path.isabs(path) else path
    with open(abs_path, "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return config


def main(project_config_path: str, run_configs_path: str, run_config_name: str, use_wandb: bool):
    # Load configs
    project_config: dict = load_config(path=project_config_path)
    run_config: dict = load_config(path=run_configs_path)[run_config_name]

    print("EXECUTING RUN:", run_config_name)
    # pprint.pprint(run_config, sort_dicts=False)
    print()

    # Train model
    train(project_config=project_config, run_config=run_config, use_wandb=use_wandb)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--project_config_path", type=str, default="project_config.yaml")
    parser.add_argument("--run_configs_path", type=str, default="run_configs.yaml")
    parser.add_argument("--run_config_name", type=str, default="SIMPLE_CONFIG_EXAMPLE_MNIST")
    parser.add_argument("--no_wandb", dest='use_wandb', action='store_false')
    parser.set_defaults(use_wandb=True)
    args = parser.parse_args()

    main(project_config_path=args.project_config_path,
         run_configs_path=args.run_configs_path,
         run_config_name=args.run_config_name,
         use_wandb=args.use_wandb)
