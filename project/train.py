from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.profiler import SimpleProfiler, AdvancedProfiler
from pytorch_lightning.loggers import WandbLogger
import yaml

# custom
from pipeline_modules.lightning_wrapper import LitModel
from project.pipeline_modules.data_modules import *
from project.pipeline_modules.callbacks import *


def train(config):

    # Init data module
    datamodule = MNISTDataModule(
        batch_size=config["hparams"]["batch_size"],
        split_ratio=config["hparams"]["split_ratio"]
    )
    datamodule.prepare_data()
    datamodule.setup()

    # Init our model
    model = LitModel(config)

    # Init wandb logger
    wandb_logger = init_wandb(config, model, datamodule)

    # Init callbacks
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
        ),
        # ExampleCallback(),
        # ConfusionMatrixLoggerCallback(),
        # UnfreezeModelCallback(wait_epochs=5),
        # SaveModelOnnxCallback(datamodule=datamodule, save_dir=wandb_logger.save_dir)
    ]

    # Init trainer
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        gpus=config["num_of_gpus"],
        max_epochs=config["hparams"]["max_epochs"],
        resume_from_checkpoint=config["resume"]["ckpt_path"] if config["resume"]["resume_from_ckpt"] else None,
        accumulate_grad_batches=config["hparams"]["accumulate_grad_batches"],
        gradient_clip_val=config["hparams"]["gradient_clip_val"],
        progress_bar_refresh_rate=config["printing"]["progress_bar_refresh_rate"],
        profiler=SimpleProfiler() if config["printing"]["profiler"] else None,
        weights_summary=config["printing"]["weights_summary"],
        num_sanity_val_steps=3,
        # fast_dev_run=True,
        # min_epochs=10,
        # limit_train_batches=0.01
        # limit_val_batches=0.01
        # limit_test_batches=0.01
        # auto_scale_batch_size="power",
        # amp_backend='apex',
        # precision=16,
    )

    # Test before training
    # trainer.test(model=model, datamodule=datamodule)

    # Save randomly initialized model
    # trainer.save_checkpoint("random.ckpt")

    # Train the model ⚡
    trainer.fit(model=model, datamodule=datamodule)

    # Evaluate model on test set
    trainer.test()


def init_wandb(config, model, dataloader):
    wandb_logger = WandbLogger(
        project=config["loggers"]["wandb"]["project"],
        job_type=config["loggers"]["wandb"]["job_type"],
        tags=config["loggers"]["wandb"]["tags"],
        entity=config["loggers"]["wandb"]["team"],
        id=config["resume"]["wandb_run_id"] if config["resume"]["resume_from_ckpt"] else None,
        log_model=config["loggers"]["wandb"]["log_model"],
        offline=config["loggers"]["wandb"]["offline"]
    )
    wandb_logger.watch(model.model, log=None)
    wandb_logger.log_hyperparams({
        "model_name": model.model.__class__.__name__,
        "dataset_name": dataloader.__class__.__name__,
        "optimizer": model.configure_optimizers().__class__.__name__,
        "train_size": len(dataloader.data_train) if dataloader.data_train is not None else 0,
        "val_size": len(dataloader.data_val) if dataloader.data_train is not None else 0,
        "test_size": len(dataloader.data_test) if dataloader.data_train is not None else 0,
        "input_dims": dataloader.input_dims,
        "input_size": dataloader.input_size,
    })
    # download model from a specific wandb run
    # wandb.restore('model-best.h5', run_path="kino/some_project/a1b2c3d")
    return wandb_logger


def load_config():
    with open("config.yaml", "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return config


if __name__ == "__main__":
    train(config=load_config())
