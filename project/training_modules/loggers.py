from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger, CometLogger, MLFlowLogger, NeptuneLogger
import os


def get_wandb_logger(train_config, model_config, lit_model, datamodule):
    wandb_logger = WandbLogger(
        project=train_config["loggers"]["wandb"]["project"],
        job_type=train_config["loggers"]["wandb"]["job_type"],
        tags=train_config["loggers"]["wandb"]["tags"],
        entity=train_config["loggers"]["wandb"]["team"],
        id=model_config["resume_training"]["wandb"]["wandb_run_id"]
        if model_config["resume_training"]["wandb"]["resume_wandb_run"] else None,
        log_model=train_config["loggers"]["wandb"]["log_model"],
        offline=train_config["loggers"]["wandb"]["offline"],
        save_dir="logs/",
        save_code=False
    )
    if not os.path.exists("logs/"):
        os.mkdir("logs/")
    wandb_logger.watch(lit_model.model, log=None)
    wandb_logger.log_hyperparams({
        "model_name": lit_model.model.__class__.__name__,
        "datamodule_name": datamodule.__class__.__name__,
        "optimizer": lit_model.configure_optimizers().__class__.__name__,
        "train_size": len(datamodule.data_train) if datamodule.data_train is not None else 0,
        "val_size": len(datamodule.data_val) if datamodule.data_train is not None else 0,
        "test_size": len(datamodule.data_test) if datamodule.data_train is not None else 0,
        "input_dims": datamodule.input_dims,
        "input_size": datamodule.input_size,
    })
    # download model from a specific wandb run
    # wandb.restore('model-best.h5', run_path="kino/some_project/a1b2c3d")
    return wandb_logger


def get_tensorboard_loggger():
    """TODO"""
    return None
