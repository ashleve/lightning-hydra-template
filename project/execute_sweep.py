# THIS FILE SHOULDN't BE EXECUTED DIRECTLY, IT SHOULD BE RUN BY WANDB SWEEP AGENT!
import wandb
import train


# Choose which run config to optimize
RUN_CONFIG = "MNIST_CLASSIFIER_V1"

# Load project config
project_conf = train.load_config("project_config.yaml")
run_conf = train.load_config("run_configs.yaml")[RUN_CONFIG]

# Initialize wandb
wandb.init()

# Replace project config hyperparameters with the ones loaded from wandb sweep server
sweep_hparams = wandb.Config._as_dict(wandb.config)
for key, value in sweep_hparams.items():
    if key == "_wandb":
        continue
    elif key in run_conf["trainer"].keys():
        run_conf["trainer"][key] = value
    elif key in run_conf["model"].keys():
        run_conf["model"][key] = value
    elif key in run_conf["dataset"].keys():
        run_conf["dataset"][key] = value

# Execute run
train.train(project_config=project_conf, run_config=run_conf)
