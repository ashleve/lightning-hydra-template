import wandb
import train


# THIS FILE SHOULDN't BE EXECUTED DIRECTLY, IT SHOULD BE RUN BY WANDB SWEEP AGENT!

# Load project config
conf = train.load_config()

# Choose model for hyperparameter search
model_conf = conf["model_configs"]["simple_mnist_classifier_v1"]

# Initialize wandb
wandb.init()

# Replace project config hyperparameters with the ones loaded from wandb sweep server
sweep_hparams = wandb.Config._as_dict(wandb.config)
for key, value in sweep_hparams.items():
    if key != "_wandb" and not key.startswith("dataset/"):
        model_conf[key] = value

# Execute run
train.train(config=conf, model_config=model_conf)
