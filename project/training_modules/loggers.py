from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger, CometLogger, MLFlowLogger, NeptuneLogger
import inspect
import os


def get_wandb_logger(config, lit_model, datamodule):
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
        "parent_folder": os.path.basename(os.path.dirname((inspect.getfile(lit_model.__class__)))),
        "datamodule_name": datamodule.__class__.__name__,
        "batch_size": datamodule.batch_size,
        "optimizer": lit_model.configure_optimizers().__class__.__name__,
        "train_size": len(datamodule.data_train) if datamodule.data_train is not None else 0,
        "val_size": len(datamodule.data_val) if datamodule.data_train is not None else 0,
        "test_size": len(datamodule.data_test) if datamodule.data_train is not None else 0,
    })
    return wandb_logger


def get_tensorboard_loggger():
    """TODO"""
    return None
