from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.profiler import SimpleProfiler
import yaml

# training modules
from training_modules.datamodules import *
from training_modules.callbacks import *
from training_modules.loggers import *

# custom models
from models import simple_mnist_classifier
from models import transfer_learning_img_classifier


def train(config):

    # CHOOSE MODEL AND DATASET HERE!!
    MODEL = simple_mnist_classifier
    DATASET = MNISTDataModule
    MODEL_PARAMS = config["models"]["simple_mnist_classifier"]["hparams"]
    DATASET_PARAMS = config["models"]["simple_mnist_classifier"]["datasets"]["MNISTDataModule"]

    # Init model
    lit_model = MODEL.LitModel(hparams=MODEL_PARAMS)

    # Init data
    datamodule = DATASET(**DATASET_PARAMS, transforms=MODEL.train_preprocess)
    datamodule.prepare_data()
    datamodule.setup()

    # Init logger
    logger = get_wandb_logger(config, lit_model, datamodule)

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
        # SaveCodeToWandbCallback(wandb_save_dir=logger.save_dir),
        # SaveOnnxModelToWandbCallback(datamodule=datamodule, save_dir=logger.save_dir)
    ]

    # Init trainer
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        gpus=config["num_of_gpus"],

        resume_from_checkpoint=config["resume_training"]["lightning_ckpt"]["ckpt_path"]
        if config["resume_training"]["lightning_ckpt"]["resume_from_ckpt"] else None,

        # model related:
        max_epochs=MODEL_PARAMS["max_epochs"],
        accumulate_grad_batches=MODEL_PARAMS["accumulate_grad_batches"],
        gradient_clip_val=MODEL_PARAMS["gradient_clip_val"],

        # print related:
        progress_bar_refresh_rate=config["printing"]["progress_bar_refresh_rate"],
        profiler=SimpleProfiler() if config["printing"]["profiler"] else None,
        weights_summary=config["printing"]["weights_summary"],

        # these are mostly for debugging:
        fast_dev_run=False,
        num_sanity_val_steps=3,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        limit_test_batches=1.0,
        val_check_interval=1.0,

        default_root_dir="logs/lightning_logs",
    )

    # Test before training
    # trainer.test(model=model, datamodule=datamodule)

    # Save randomly initialized model
    # trainer.save_checkpoint("random.ckpt")

    # Train the model ⚡
    trainer.fit(model=lit_model, datamodule=datamodule)

    # Evaluate model on test set
    trainer.test()


def load_config():
    with open("config.yaml", "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return config


if __name__ == "__main__":
    train(config=load_config())
