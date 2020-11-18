from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger, CometLogger, MLFlowLogger, NeptuneLogger
import os


def init_wandb(config, lit_model, datamodule):
    wandb_logger = WandbLogger(
        project=config["loggers"]["wandb"]["project"],
        job_type=config["loggers"]["wandb"]["job_type"],
        tags=config["loggers"]["wandb"]["tags"],
        entity=config["loggers"]["wandb"]["team"],
        id=config["resume"]["wandb_run_id"] if config["resume"]["resume_from_ckpt"] else None,
        log_model=config["loggers"]["wandb"]["log_model"],
        offline=config["loggers"]["wandb"]["offline"],
        save_dir="logs/"
    )
    os.mkdir("logs/") if not os.path.exists("logs/") else None
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
