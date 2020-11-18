from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.profiler import SimpleProfiler
import yaml

# custom
from pipeline_modules.lightning_wrapper import LitModel
from pipeline_modules.logger_initializers import *
from pipeline_modules.data_modules import *
from pipeline_modules.callbacks import *


def train(config):
    # Init data module
    datamodule = MNISTDataModule(
        batch_size=config["hparams"]["batch_size"],
        split_ratio=config["hparams"]["split_ratio"]
    )
    datamodule.prepare_data()
    datamodule.setup()

    # Init our model
    model = LitModel(hparams=config["hparams"])

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
        # MetricsHeatmapLoggerCallback(),
        # UnfreezeModelCallback(wait_epochs=5),
        # ImagePredictionLoggerCallback(datamodule=datamodule),
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
        default_root_dir="logs/lightning_logs"
        # fast_dev_run=True,
        # min_epochs=10,
        # limit_train_batches=0.1
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


def load_config():
    with open("config.yaml", "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return config


if __name__ == "__main__":
    train(config=load_config())
