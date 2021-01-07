# THIS FILE IS FOR HYPERPARAMETER SEARCH WITH Weights&Biases
# IT SHOULDN't BE EXECUTED DIRECTLY, IT SHOULD BE RUN BY WANDB SWEEP AGENT!

import wandb
import train


# Choose which run config to optimize
RUN_CONFIG_NAME = "SIMPLE_CONFIG_EXAMPLE_MNIST"

# Load configs
project_config = train.load_config("config.yaml")
run_config = train.load_config("run_configs.yaml")[RUN_CONFIG_NAME]

# Initialize wandb
wandb.init()

# Replace run config hyperparameters with the ones loaded from wandb sweep server
sweep_hparams = wandb.Config._as_dict(wandb.config)
for key, value in sweep_hparams.items():
    if key == "_wandb":
        continue
    elif key in run_config["trainer"]["args"]:
        run_config["trainer"]["args"][key] = value
    elif key in run_config["model"]["hparams"]:
        run_config["model"]["hparams"][key] = value
    elif key in run_config["datamodule"]["hparams"]:
        run_config["datamodule"]["hparams"][key] = value

# Execute run
train.train(project_config=project_config, run_config=run_config, use_wandb=True)
