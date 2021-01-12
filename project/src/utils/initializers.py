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
    class_path, obj_name = class_path.rsplit('.', 1)
    module = importlib.import_module(class_path)
    assert hasattr(module, obj_name), \
        f'Object `{obj_name}` cannot be loaded from `{class_path}`.'
    return getattr(module, obj_name)


def init_object(class_path, args) -> object:
    return load_class(class_path)(**args)


def init_model(model_config: dict, optimizer_config: dict) -> pl.LightningModule:
    """
    Load LightningModule from path specified in config.
    """
    model_class = model_config["class"]
    LitModel = load_class(model_class)
    assert issubclass(LitModel, pl.LightningModule), \
        f"Specified model class `{model_class}` is not a subclass of `LightningModule`."
    return LitModel(model_config=model_config, optimizer_config=optimizer_config)


def init_datamodule(datamodule_config: dict, data_dir: str) -> pl.LightningDataModule:
    """
    Load LightningDataModule from path specified config.
    """
    datamodule_class = datamodule_config["class"]
    datamodule_args = datamodule_config["args"]
    DataModule = load_class(datamodule_class)
    assert issubclass(DataModule, pl.LightningDataModule), \
        f"Specified datamodule class `{datamodule_class}` is not a subclass of `LightningDataModule`"
    datamodule = DataModule(data_dir=data_dir, **datamodule_args)
    datamodule.prepare_data()
    datamodule.setup()
    return datamodule


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


def init_callbacks(callbacks_config: dict) -> List[pl.Callback]:
    """
    Initialize callbacks specified in config.
    """
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
    """
    Initialize loggers specified in config.
    """
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
        if hasattr(model, 'model'):
            wandb_logger.watch(model.model, log=None)
        else:
            wandb_logger.watch(model, log=None)


def log_hparams(loggers: List[pl.loggers.LightningLoggerBase], hparams: dict):
    for logger in loggers:
        logger.log_hyperparams(hparams)


def log_extra_hparams(loggers: List[pl.loggers.LightningLoggerBase],
                      config: dict,
                      model: pl.LightningModule,
                      datamodule: pl.LightningDataModule,
                      callbacks: List[pl.callbacks.Callback]):
    if "extra_logs" not in config:
        return

    extra_logs = config["extra_logs"]

    if "log_seeds" in extra_logs and extra_logs["log_seeds"] and "seeds" in config:
        log_hparams(loggers, config["seeds"])

    if "log_trainer_args" in extra_logs and extra_logs["log_trainer_args"]:
        log_hparams(loggers, config["trainer"]["args"])
    if "log_model_args" in extra_logs and extra_logs["log_model_args"]:
        log_hparams(loggers, config["model"]["args"])
    if "log_optimizer_args" in extra_logs and extra_logs["log_optimizer_args"]:
        log_hparams(loggers, config["optimizer"]["args"])
    if "log_datamodule_args" in extra_logs and extra_logs["log_datamodule_args"]:
        log_hparams(loggers, config["datamodule"]["args"])

    if "log_model_class" in extra_logs and extra_logs["log_model_class"]:
        log_hparams(loggers, {"_class_model": config["model"]["class"]})
    if "log_optimizer_class" in extra_logs and extra_logs["log_optimizer_class"]:
        log_hparams(loggers, {"_class_optimizer": config["optimizer"]["class"]})
    if "log_datamodule_class" in extra_logs and extra_logs["log_datamodule_class"]:
        log_hparams(loggers, {"_class_datamodule": config["datamodule"]["class"]})

    if "log_train_val_test_sizes" in extra_logs and extra_logs["log_train_val_test_sizes"]:
        hparams = {}
        if hasattr(datamodule, 'data_train') and datamodule.data_train is not None:
            hparams["train_size"] = len(datamodule.data_train)
        if hasattr(datamodule, 'data_val') and datamodule.data_val is not None:
            hparams["val_size"] = len(datamodule.data_val)
        if hasattr(datamodule, 'data_test') and datamodule.data_test is not None:
            hparams["test_size"] = len(datamodule.data_test)
        log_hparams(loggers, hparams)

    if "log_model_architecture_class" in extra_logs and extra_logs["log_model_architecture_class"]:
        if hasattr(model, "model"):
            obj = model.model
            log_hparams(loggers, {"_class_model_architecture": obj.__module__ + "." + obj.__class__.__name__})


def show_init_info(model, datamodule, callbacks, loggers):
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


def auto_find_lr(trainer, model, datamodule, loggers):
    lr_finder = trainer.tuner.lr_find(model=model, datamodule=datamodule)
    new_lr = lr_finder.suggestion()

    # Save lr plot
    fig = lr_finder.plot(suggest=True)
    fig.savefig("lr_loss_curve.jpg")

    # Set new lr
    model.hparams.lr = new_lr
    log_hparams(loggers, {"lr": new_lr})


def show_config(config: DictConfig):
    log.info(f"\n{OmegaConf.to_yaml(config, resolve=True)}")


def validate_config(config: dict):
    # TODO
    pass


def validate_obj_config(obj_config: dict):
    # TODO
    pass
