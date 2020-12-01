from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import importlib
import os


def init_lit_model(hparams):
    """Load LitModel from folder specified in run config."""
    module_path = "models." + hparams["model_folder"] + ".lightning_module"
    lit_model = importlib.import_module(module_path).LitModel(hparams=hparams)
    return lit_model


def init_data_module(hparams):
    """Load DataModule from folder specified in run config."""
    module_path = "data_modules." + hparams["datamodule_folder"] + ".datamodule"
    datamodule = importlib.import_module(module_path).DataModule(hparams=hparams)
    datamodule.prepare_data()
    datamodule.setup()
    return datamodule


def init_main_callbacks(project_config):
    """Initialize EarlyStopping callback and ModelCheckpoint callback."""
    callbacks = [
        EarlyStopping(
            monitor=project_config["callbacks"]["early_stop"]["monitor"],
            patience=project_config["callbacks"]["early_stop"]["patience"],
            mode=project_config["callbacks"]["early_stop"]["mode"],
        ),
        ModelCheckpoint(
            monitor=project_config["callbacks"]["checkpoint"]["monitor"],
            save_top_k=project_config["callbacks"]["checkpoint"]["save_top_k"],
            mode=project_config["callbacks"]["checkpoint"]["mode"],
            save_last=project_config["callbacks"]["checkpoint"]["save_last"],
        )
    ]
    return callbacks


def init_wandb_logger(config, run_config, lit_model, datamodule):
    """Initialize Weights&Biases logger."""
    resume_from_checkpoint = run_config.get("resume_training", {}).get("resume_from_checkpoint", None)
    wandb_run_id = run_config.get("resume_training", {}).get("wandb_run_id", None)
    wandb_logger = WandbLogger(
        project=config["loggers"]["wandb"]["project"],
        entity=config["loggers"]["wandb"]["team"],
        log_model=config["loggers"]["wandb"]["log_model"],
        offline=config["loggers"]["wandb"]["offline"],

        group=run_config.get("wandb", {}).get("group", None),
        job_type=run_config.get("wandb", {}).get("job_type", "train"),
        tags=run_config.get("wandb", {}).get("tags", []),

        # resume run only if ckpt was set in the run config
        id=wandb_run_id
        if resume_from_checkpoint != "None" and wandb_run_id != "None"
        and resume_from_checkpoint != "False" and wandb_run_id != "False"
        and resume_from_checkpoint is not False and wandb_run_id is not False
        else None,

        save_dir="logs/",
        save_code=False
    )
    if not os.path.exists("logs/"):
        os.mkdir("logs/")
    if hasattr(lit_model, 'model'):
        wandb_logger.watch(lit_model.model, log=None)
    else:
        wandb_logger.watch(lit_model, log=None)
    wandb_logger.log_hyperparams({
        "model_name": lit_model.model.__class__.__name__,
        "optimizer": lit_model.configure_optimizers().__class__.__name__,
        "train_size": len(datamodule.data_train) if datamodule.data_train is not None else 0,
        "val_size": len(datamodule.data_val) if datamodule.data_val is not None else 0,
        "test_size": len(datamodule.data_test) if datamodule.data_test is not None else 0,
    })
    wandb_logger.log_hyperparams(run_config["trainer"])
    wandb_logger.log_hyperparams(run_config["model"])
    wandb_logger.log_hyperparams(run_config["dataset"])
    return wandb_logger


def init_tensorboard_loggger():
    """TODO"""
    return None
