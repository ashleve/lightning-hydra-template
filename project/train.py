from pytorch_lightning.profiler import SimpleProfiler
import yaml

# lightning modules
from lightning_modules.init_utils import *
from lightning_modules.callbacks import *


def train(config, model_config):

    # Init lightning model
    lit_model = init_lit_model(model_config)

    # Init lightning datamodule
    datamodule = init_datamodule(model_config)

    # Init logger
    logger = init_wandb_logger(config, lit_model, datamodule)

    # Init callbacks
    callbacks = init_main_callbacks(config)
    callbacks.extend([
        # MetricsHeatmapLoggerCallback(),
        # UnfreezeModelCallback(wait_epochs=5),
        # ImagePredictionLoggerCallback(datamodule=datamodule),
        SaveCodeToWandbCallback(wandb_save_dir=logger.save_dir, lit_model=lit_model),
        # SaveOnnxModelToWandbCallback(datamodule=datamodule, save_dir=logger.save_dir)
    ])

    # Init trainer
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        gpus=config["num_of_gpus"],

        resume_from_checkpoint=config["resume_training"]["lightning_ckpt"]["ckpt_path"]
        if config["resume_training"]["lightning_ckpt"]["resume_from_ckpt"] else None,

        # model related:
        max_epochs=model_config["max_epochs"],
        accumulate_grad_batches=model_config["accumulate_grad_batches"],
        gradient_clip_val=model_config["gradient_clip_val"],

        # print related:
        progress_bar_refresh_rate=config["printing"]["progress_bar_refresh_rate"],
        profiler=SimpleProfiler() if config["printing"]["profiler"] else None,
        weights_summary=config["printing"]["weights_summary"],

        # these are mostly for debugging (read TIPS.md for explanation):
        fast_dev_run=False,
        num_sanity_val_steps=3,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        limit_test_batches=1.0,
        val_check_interval=1.0,

        default_root_dir="logs/lightning_logs",
    )

    # Test before training
    # trainer.test(model=lit_model, datamodule=datamodule)

    # Save checkpoint
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
    conf = load_config()

    # --------------------------------------- CHOOSE YOUR MODEL HEREEE ↓↓↓ --------------------------------------- #

    model_conf = conf["model_configs"]["simple_mnist_classifier_v1"]
    # model_conf = conf["model_configs"]["transfer_learning_cifar10_classifier_v1"]

    # --------------------------------------- CHOOSE YOUR MODEL HEREEE ↑↑↑ --------------------------------------- #

    train(config=conf, model_config=model_conf)
