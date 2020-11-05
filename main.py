# regular imports
import yaml

# pytorch lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.profiler import SimpleProfiler, AdvancedProfiler

# wandb
import wandb
from pytorch_lightning.loggers import WandbLogger

# custom utils
from utils.data_modules import MNISTDataModule
from utils.lightning_wrapper import LitModel
from utils.callbacks import ExampleCallback


def main(config):
    # Init our model
    model = LitModel(config=config)

    # Init data loader
    dataloader = MNISTDataModule(batch_size=128)
    dataloader.prepare_data()
    dataloader.setup()

    # Init wandb logger
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
    })

    # Init callbacks
    callbacks = [
        # ExampleCallback(),
        EarlyStopping(
            monitor=config["callbacks"]["early_stop"]["monitor"],
            patience=config["callbacks"]["early_stop"]["patience"],
            mode=config["callbacks"]["early_stop"]["mode"],
            verbose=False
        ),
        ModelCheckpoint(
            monitor=config["callbacks"]["checkpoint"]["monitor"],
            save_top_k=config["callbacks"]["checkpoint"]["save_top_k"],
            mode=config["callbacks"]["checkpoint"]["mode"],
            save_last=config["callbacks"]["checkpoint"]["save_last"],
            verbose=False
        )
    ]

    # Init trainer
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        gpus=config["hparams"]["num_of_gpus"],
        max_epochs=config["hparams"]["max_epochs"],
        resume_from_checkpoint=config["resume"]["ckpt_path"] if config["resume"]["resume_from_ckpt"] else None,
        accumulate_grad_batches=config["hparams"]["accumulate_grad_batches"],
        gradient_clip_val=config["hparams"]["gradient_clip_val"],
        progress_bar_refresh_rate=config["printing"]["progress_bar_refresh_rate"],
        profiler=SimpleProfiler() if config["printing"]["profiler"] else None,
        weights_summary=config["printing"]["weights_summary"],
        # fast_dev_run=True,
        # limit_train_batches=0.01
        # limit_val_batches=0.01
        # limit_test_batches=0.01
        # auto_scale_batch_size="power",
        # min_epochs=10,
        # amp_backend='apex',
        # precision=16,
    )

    # Train the model ⚡
    trainer.fit(model=model, datamodule=dataloader)

    # Evaluate model on test set
    trainer.test()

    # Close wandb run
    wandb.finish()


def load_config():
    with open("config.yaml", "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return config


if __name__ == "__main__":
    main(config=load_config())
