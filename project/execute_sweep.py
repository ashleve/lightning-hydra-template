import wandb
import train


# Load project config
config = train.load_config()

# Initialize wandb
wandb.init()

# Replace project config hyperparameters with the ones loaded from wandb sweep server
sweep_hparams = wandb.Config._as_dict(wandb.config)
for key, value in sweep_hparams.items():
    if key != "_wandb":
        config["hparams"][key] = value

# Execute run
train.train(config)
