from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger, CometLogger, MLFlowLogger, NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from lightning_modules.data_modules import datamodules, transforms
import importlib


def init_lit_model(model_config):
    module_path = "models." + model_config["model_folder_name"] + ".lightning_module"
    return importlib.import_module(module_path).LitModel(hparams=model_config)


def init_datamodule(model_config):
    datamodule_class = getattr(datamodules, model_config["dataset"]["class_name"])
    train_transforms = getattr(transforms, model_config["dataset"]["train_transforms"])
    datamodule = datamodule_class(
        transforms=train_transforms,
        batch_size=model_config["batch_size"],
        **model_config["dataset"]
    )
    datamodule.prepare_data()
    datamodule.setup()
    return datamodule


def init_main_callbacks(config):
    callbacks = [
        EarlyStopping(
            monitor=config["callbacks"]["early_stop"]["monitor"],
            patience=config["callbacks"]["early_stop"]["patience"],
            mode=config["callbacks"]["early_stop"]["mode"],
        ),
        ModelCheckpoint(
            monitor=config["callbacks"]["checkpoint"]["monitor"],
            save_top_k=config["callbacks"]["checkpoint"]["save_top_k"],
            mode=config["callbacks"]["checkpoint"]["mode"],
            save_last=config["callbacks"]["checkpoint"]["save_last"],
        )
    ]
    return callbacks


def init_wandb_logger(config, lit_model, datamodule):
    wandb_logger = WandbLogger(
        project=config["loggers"]["wandb"]["project"],
        job_type=config["loggers"]["wandb"]["job_type"],
        tags=config["loggers"]["wandb"]["tags"],
        entity=config["loggers"]["wandb"]["team"],
        id=config["resume_training"]["wandb"]["wandb_run_id"]
        if config["resume_training"]["wandb"]["resume_wandb_run"] else None,
        log_model=config["loggers"]["wandb"]["log_model"],
        offline=config["loggers"]["wandb"]["offline"],
        save_dir="logs/",
        save_code=False
    )
    wandb_logger.watch(lit_model.model, log=None)
    wandb_logger.log_hyperparams({
        "model_name": lit_model.model.__class__.__name__,
        "optimizer": lit_model.configure_optimizers().__class__.__name__,
        "train_size": len(datamodule.data_train) if datamodule.data_train is not None else 0,
        "val_size": len(datamodule.data_val) if datamodule.data_train is not None else 0,
        "test_size": len(datamodule.data_test) if datamodule.data_train is not None else 0,
    })
    return wandb_logger


def init_tensorboard_loggger():
    """TODO"""
    return None
