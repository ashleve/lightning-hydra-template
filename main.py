# pytorch lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.profiler import SimpleProfiler, AdvancedProfiler

# yaml
import yaml

# wandb
import wandb
from pytorch_lightning.loggers import WandbLogger

# custom utils
from utils.lightning_wrapper import LitModel
from utils.data_modules import MNISTDataModule
from utils.callbacks import ExampleCallback


def init_wandb(config, model, dataloader):
    wandb_logger = WandbLogger(
        project=config["loggers"]["wandb"]["project"],
        job_type=config["loggers"]["wandb"]["job_type"],
        tags=config["loggers"]["wandb"]["tags"],
        entity=config["loggers"]["wandb"]["team"],
        log_model=True,
        offline=False
    )
    wandb_logger.watch(model.model, log='all')
    wandb_logger.log_hyperparams({
        "model_name": model.model.__class__.__name__,
        "dataset_name": dataloader.__class__.__name__,
        "optimizer": model.configure_optimizers().__class__.__name__,
        "train_size": len(dataloader.data_train),
        "val_size": len(dataloader.data_val),
        "test_size": len(dataloader.data_test),
        "input_dims": dataloader.dims,
    })
    return wandb_logger


def main(config):
    # Init our model
    model = LitModel(config)

    # Init data loader
    dataloader = MNISTDataModule(batch_size=config["hparams"]["batch_size"])
    dataloader.prepare_data()
    dataloader.setup()

    # Init wandb logger
    wandb_logger = init_wandb(config, model, dataloader)

    # Init callbacks
    callbacks = [
        # ExampleCallback(),
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
        # fast_dev_run=True,
        # min_epochs=10,
        # limit_train_batches=0.01
        # limit_val_batches=0.01
        # limit_test_batches=0.01
        # auto_scale_batch_size="power",
        # amp_backend='apex',
        # precision=16,
    )

    # Train the model ⚡
    trainer.fit(model=model, datamodule=dataloader)

    # Evaluate model on test set
    trainer.test()


def load_config():
    with open("config.yaml", "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return config


if __name__ == "__main__":
    main(config=load_config())
