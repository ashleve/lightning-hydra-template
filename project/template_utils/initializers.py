# pytorch lightning imports
import pytorch_lightning as pl

# normal imports
from typing import List, Tuple
import importlib.util
import os

# template utils imports
from pytorch_lightning.loggers import CSVLogger, WandbLogger, CometLogger, TestTubeLogger, TensorBoardLogger
import torch


def format_path(path: str, base_dir: str):
    """Convert path to absolute relatively to base_dir and normalize it."""
    if not os.path.isabs(path):
        path = os.path.join(base_dir, path)
    return os.path.normpath(path)


def format_config_paths(config: dict, base_dir: str) -> dict:
    config["paths"]["data_dir"] = format_path(config["paths"]["data_dir"], base_dir)
    config["paths"]["logs_dir"] = format_path(config["paths"]["logs_dir"], base_dir)
    return config


def load_object(obj_path):
    obj_path, obj_name = obj_path.rsplit('.', 1)
    module = importlib.import_module(obj_path)
    assert hasattr(module, obj_name), \
        f'Object `{obj_name}` cannot be loaded from `{obj_path}`.'
    return getattr(module, obj_name)


def init_object(obj_path, args):
    return load_object(obj_path)(**args)


def init_model(model_config: dict, optimizer_config: dict) -> pl.LightningModule:
    """
    Load LightningModule from path specified in config.
    """
    model_class = model_config["class"]
    model_args = model_config["args"]
    LitModel = load_object(model_class)
    assert issubclass(LitModel, pl.LightningModule), \
        f"Specified model class `{model_class}` is not a subclass of `LightningModule`."
    return LitModel(optimizer_config, **model_args)


def init_datamodule(datamodule_config: dict, data_dir: str) -> pl.LightningDataModule:
    """
    Load LightningDataModule from path specified config.
    """
    datamodule_class = datamodule_config["class"]
    datamodule_args = datamodule_config["args"]
    DataModule = load_object(datamodule_class)
    assert issubclass(DataModule, pl.LightningDataModule), \
        f"Specified datamodule class `{datamodule_class}` is not a subclass of `LightningDataModule`"
    datamodule = DataModule(data_dir=data_dir, **datamodule_args)
    datamodule.prepare_data()
    datamodule.setup()
    return datamodule


def init_optimizer(optimizer_config, model):
    optimizer_class = optimizer_config["class"]
    optimizer_args = optimizer_config["args"]
    Optim = load_object(optimizer_class)
    assert issubclass(Optim, torch.optim.Optimizer), \
        f"Specified optimizer class `{optimizer_class}` is not a subclass of `torch.optim.Optimizer`."
    return Optim(model.parameters(), **optimizer_args)


def init_trainer(trainer_config: dict,
                 callbacks: List[pl.Callback],
                 loggers: List[pl.loggers.LightningLoggerBase]) -> pl.Trainer:
    """
    Initialize PyTorch Lightning Trainer.
    """
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=loggers,
        **trainer_config["args"]
    )
    return trainer


def init_callbacks(config: dict) -> List[pl.Callback]:
    """
    Initialize callbacks specified in config.
    """
    callbacks_config = config.get("callbacks", None)
    callbacks = []

    if not callbacks_config:
        return callbacks

    for callback_name, callback_conf in callbacks_config.items():
        callback_class = callback_conf.get("class", None)
        if callback_class:
            callback_obj = init_object(callback_class, callback_conf.get("args", {}))
            callbacks.append(callback_obj)

    return callbacks


def init_loggers(config: dict,
                 model: pl.LightningModule,
                 datamodule: pl.LightningDataModule) -> List[pl.loggers.LightningLoggerBase]:
    """
    Initialize loggers specified in config.
    """
    loggers_config = config.get("loggers", None)
    loggers = []

    if not loggers_config:
        return loggers

    for logger_name, logger_conf in loggers_config.items():
        logger_class = loggers_config.get("class", None)
        if logger_class:
            if logger_class == "pytorch_lightning.loggers.WandbLogger":
                logger_obj = init_wandb(config, model, datamodule)
            else:
                logger_obj = init_object(logger_class, logger_conf.get("args", {}))
            loggers.append(logger_obj)

    return loggers


def init_wandb(config: dict,
               model: pl.LightningModule,
               datamodule: pl.LightningDataModule) -> pl.loggers.WandbLogger:
    """
    Initialize Weights&Biases logger.
    """
    # with this line wandb will throw an error if the run to be resumed does not exist yet
    # instead of auto-creating a new run
    os.environ["WANDB_RESUME"] = "must"

    wandb_config = config["loggers"]["wandb"]
    wandb_logger = WandbLogger(**wandb_config["args"])

    # make save dir path if it doesn't exists to prevent throwing errors by wandb
    if "save_dir" in wandb_config["args"] and not os.path.exists(wandb_config["args"]["save_dir"]):
        os.makedirs(wandb_config["args"]["save_dir"])

    if hasattr(model, 'model'):
        if wandb_config["extra_options"]["log_gradients"]:
            wandb_logger.watch(model.model, log='gradients')
        else:
            wandb_logger.watch(model.model, log=None)
        wandb_logger.log_hyperparams({"architecture": model.model.__class__.__name__})
    else:
        if wandb_config["extra_options"]["log_gradients"]:
            wandb_logger.watch(model, log='gradients')
        else:
            wandb_logger.watch(model, log=None)

    if wandb_config["extra_options"]["log_train_val_test_sizes"]:
        wandb_logger.log_hyperparams({
            "train_size": len(datamodule.data_train)
            if hasattr(datamodule, 'data_train') and datamodule.data_train is not None else 0,
            "val_size": len(datamodule.data_val)
            if hasattr(datamodule, 'data_val') and datamodule.data_val is not None else 0,
            "test_size": len(datamodule.data_test)
            if hasattr(datamodule, 'data_test') and datamodule.data_test is not None else 0,
        })

    if wandb_config["extra_options"]["log_model_args"]:
        wandb_logger.log_hyperparams(config["model"]["args"])
    if wandb_config["extra_options"]["log_datamodule_args"]:
        wandb_logger.log_hyperparams(config["datamodule"]["args"])
    if wandb_config["extra_options"]["log_optimizer_args"]:
        wandb_logger.log_hyperparams(config["optimizer"]["args"])
    if wandb_config["extra_options"]["log_trainer_args"]:
        wandb_logger.log_hyperparams(config["trainer"]["args"])

    if wandb_config["extra_options"]["log_model_class_name"]:
        wandb_logger.log_hyperparams({"model_class": config["model"]["class"]})
    if wandb_config["extra_options"]["log_datamodule_class_name"]:
        wandb_logger.log_hyperparams({"datamodule_class": config["datamodule"]["class"]})
    if wandb_config["extra_options"]["log_optimizer_class_name"]:
        wandb_logger.log_hyperparams({"optimizer_class": config["optimizer"]["class"]})

    return wandb_logger
