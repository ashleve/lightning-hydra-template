# pytorch lightning imports
from pytorch_lightning.loggers.wandb import WandbLogger
import pytorch_lightning as pl

# hydra imports
from omegaconf import DictConfig, OmegaConf

# normal imports
from typing import List
import importlib
import logging

log = logging.getLogger(__name__)


def load_class(class_path) -> type:
    """Load class from a given path."""
    class_path, obj_name = class_path.rsplit('.', 1)
    module = importlib.import_module(class_path)
    assert hasattr(module, obj_name), \
        f'Object `{obj_name}` cannot be loaded from `{class_path}`.'
    return getattr(module, obj_name)


def init_object(class_path, args) -> object:
    """Initialize an object from a given class path and args."""
    return load_class(class_path)(**args)


def init_model(model_config: dict) -> pl.LightningModule:
    """Initialize LightningModule from a given model config."""
    model_class = model_config["class"]
    model_args = model_config["args"]
    LitModelClass = load_class(model_class)
    return LitModelClass(**model_args)


def init_datamodule(datamodule_config: dict) -> pl.LightningDataModule:
    """Initialize LightningDataModule from a given datamodule config."""
    datamodule_class = datamodule_config["class"]
    datamodule_args = datamodule_config["args"]
    DataModuleClass = load_class(datamodule_class)
    datamodule = DataModuleClass(**datamodule_args)
    datamodule.prepare_data()
    datamodule.setup()
    return datamodule


def init_trainer(trainer_config: dict,
                 callbacks: List[pl.Callback],
                 loggers: List[pl.loggers.LightningLoggerBase]) -> pl.Trainer:
    """Initialize PyTorch Lightning Trainer from a given trainer config."""
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=loggers,
        **trainer_config["args"]
    )
    return trainer


def init_callbacks(callbacks_config: dict) -> List[pl.Callback]:
    """Initialize callbacks from a given callbacks config."""
    callbacks = []
    for callback_name, callback_conf in callbacks_config.items():
        class_path = callback_conf.get("class", None)
        if class_path:
            callback_obj = init_object(class_path=class_path, args=callback_conf.get("args", {}))
            assert issubclass(callback_obj.__class__, pl.Callback), \
                f"Specified callback class `{class_path}` is not a subclass of `pl.Callback`."
            callbacks.append(callback_obj)
    return callbacks


def init_loggers(loggers_config: dict) -> List[pl.loggers.LightningLoggerBase]:
    """Initialize loggers from a given loggers config."""
    loggers = []
    for logger_name, logger_conf in loggers_config.items():
        class_path = logger_conf.get("class", None)
        if class_path:
            logger_obj = init_object(class_path=class_path, args=logger_conf.get("args", {}))
            assert issubclass(logger_obj.__class__, pl.loggers.LightningLoggerBase), \
                f"Specified logger class `{class_path}` is not a subclass of `pl.loggers.LightningLoggerBase`."
            loggers.append(logger_obj)
    return loggers


def get_wandb_logger(loggers: List[pl.loggers.LightningLoggerBase]) -> WandbLogger:
    for logger in loggers:
        if isinstance(logger, WandbLogger):
            return logger


def make_wandb_watch_model(loggers: List[pl.loggers.LightningLoggerBase], model: pl.LightningModule):
    wandb_logger = get_wandb_logger(loggers)
    if wandb_logger:
        if hasattr(model, 'architecture'):
            wandb_logger.watch(model.architecture, log=None)
        else:
            wandb_logger.watch(model, log=None)


def send_hparams_to_loggers(loggers: List[pl.loggers.LightningLoggerBase], hparams: dict):
    for logger in loggers:
        logger.log_hyperparams(hparams)


def log_hparams(loggers: List[pl.loggers.LightningLoggerBase],
                config: dict,
                model: pl.LightningModule,
                datamodule: pl.LightningDataModule,
                callbacks: List[pl.callbacks.Callback]):
    extra_logs = config.get("extra_logs", {})

    if extra_logs.get("save_model_class_path", None):
        hparams = {"_class_model": config["model"]["class"]}
        send_hparams_to_loggers(loggers, hparams)
    if extra_logs.get("save_optimizer_class_path", None):
        hparams = {"_class_optimizer": config["optimizer"]["class"]}
        send_hparams_to_loggers(loggers, hparams)
    if extra_logs.get("save_datamodule_class_path", None):
        hparams = {"_class_datamodule": config["datamodule"]["class"]}
        send_hparams_to_loggers(loggers, hparams)
    if extra_logs.get("save_model_architecture_class_path", None):
        if hasattr(model, "architecture"):
            obj = model.architecture
            hparams = {"_class_model_architecture": obj.__module__ + "." + obj.__class__.__name__}
            send_hparams_to_loggers(loggers, hparams)

    if extra_logs.get("save_seeds", None):
        send_hparams_to_loggers(loggers, config["seeds"])

    if extra_logs.get("save_model_args", None):
        send_hparams_to_loggers(loggers, config["model"]["args"])
    if extra_logs.get("save_datamodule_args", None):
        send_hparams_to_loggers(loggers, config["datamodule"]["args"])
    if extra_logs.get("save_optimizer_args", None):
        send_hparams_to_loggers(loggers, config["optimizer"]["args"])
    if extra_logs.get("save_trainer_args", None):
        send_hparams_to_loggers(loggers, config["trainer"]["args"])

    if extra_logs.get("save_data_train_val_test_sizes", None):
        hparams = {}
        if hasattr(datamodule, 'data_train') and datamodule.data_train is not None:
            hparams["train_size"] = len(datamodule.data_train)
        if hasattr(datamodule, 'data_val') and datamodule.data_val is not None:
            hparams["val_size"] = len(datamodule.data_val)
        if hasattr(datamodule, 'data_test') and datamodule.data_test is not None:
            hparams["test_size"] = len(datamodule.data_test)
        send_hparams_to_loggers(loggers, hparams)


def print_module_init_info(model, datamodule, callbacks, loggers):
    message = "Model initialised:" + "\n" + model.__module__ + "." + model.__class__.__name__ + "\n"
    log.info(message)

    message = "Datamodule initialised:" + "\n" + datamodule.__module__ + "." + datamodule.__class__.__name__ + "\n"
    log.info(message)

    message = "Callbacks initialised:" + "\n"
    for cb in callbacks:
        message += cb.__module__ + "." + cb.__class__.__name__ + "\n"
    log.info(message)

    message = "Loggers initialised:" + "\n"
    for logger in loggers:
        message += logger.__module__ + "." + logger.__class__.__name__ + "\n"
    log.info(message)


def print_config(config: DictConfig):
    log.info(f"\n{OmegaConf.to_yaml(config, resolve=True)}")
