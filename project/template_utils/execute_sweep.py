# THIS FILE IS FOR HYPERPARAMETER SEARCH WITH Weights&Biases
# IT SHOULDN't BE EXECUTED DIRECTLY, IT SHOULD BE RUN BY WANDB SWEEP AGENT!

# hydra imports
from omegaconf import DictConfig
import hydra

# template utils imports
import template_utils.initializers as utils

import wandb
from train import train


# Replace run config hyperparameters with the ones loaded from wandb sweep server
def replace_hparams(config, sweep_hparams):
    for key, value in sweep_hparams.items():
        if key == "_wandb":
            continue
        elif key in config["trainer"]["args"]:
            config["trainer"]["args"][key] = value
        elif key in config["model"]["args"]:
            config["model"]["args"][key] = value
        elif key in config["optimizer"]["args"]:
            config["optimizer"]["args"][key] = value
        elif key in config["datamodule"]["args"]:
            config["datamodule"]["args"][key] = value


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig) -> None:
    wandb.init()

    sweep_hparams = wandb.Config._as_dict(wandb.config)
    utils.show_config(config)  # print content of config

    train(config)
    wandb.finish()


if __name__ == "__main__":
    main()
