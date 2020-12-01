from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.loggers import WandbLogger
from argparse import ArgumentParser
import pytorch_lightning as pl
import yaml

# utils
from utils.init_utils import init_lit_model, init_data_module, init_main_callbacks, init_wandb_logger
from utils.callbacks import *


def train(project_config: dict, run_config: dict, use_wandb: bool):
    # Init PyTorch Lightning model ⚡
    lit_model: pl.LightningModule = init_lit_model(hparams=run_config["model"])

    # Init PyTorch Lightning datamodule ⚡
    datamodule: pl.LightningDataModule = init_data_module(hparams=run_config["dataset"])

    # Init Weights&Biases logger
    logger: pl.loggers.WandbLogger = init_wandb_logger(project_config, run_config, lit_model, datamodule) \
        if use_wandb else None

    # Init ModelCheckpoint and EarlyStopping callbacks
    callbacks: list = init_main_callbacks(project_config)

    # Add custom callbacks from utils/callbacks.py
    callbacks.extend([
        # MetricsHeatmapLoggerCallback(),
        # UnfreezeModelCallback(wait_epochs=5),
    ])
    if use_wandb:
        callbacks.append(
            SaveCodeToWandbCallback(wandb_save_dir=logger.save_dir, lit_model=lit_model, datamodule=datamodule),
        )

    # Get path to checkpoint you want to resume with if it was set in run config
    resume_from_checkpoint = run_config.get("resume_training", {}).get("checkpoint_path", None)

    # Init PyTorch Lightning trainer ⚡
    trainer = pl.Trainer(
        # whether to use gpu and how many
        gpus=project_config["num_of_gpus"],

        # experiment logging
        logger=logger,

        # useful callbacks
        callbacks=callbacks,

        # resume training from checkpoint if it was set in the run config
        resume_from_checkpoint=resume_from_checkpoint
        if resume_from_checkpoint != "None"
        and resume_from_checkpoint != "False"
        and resume_from_checkpoint is not False
        else None,

        # print related
        progress_bar_refresh_rate=project_config["printing"]["progress_bar_refresh_rate"],
        profiler=SimpleProfiler() if project_config["printing"]["profiler"] else None,
        weights_summary=project_config["printing"]["weights_summary"],

        # run related
        max_epochs=run_config["trainer"]["max_epochs"],
        min_epochs=run_config["trainer"].get("min_epochs", 1),
        accumulate_grad_batches=run_config["trainer"].get("accumulate_grad_batches", 1),
        gradient_clip_val=run_config["trainer"].get("gradient_clip_val", 0.5),

        # these are mostly for debugging
        fast_dev_run=run_config["trainer"].get("fast_dev_run", False),
        limit_train_batches=run_config["trainer"].get("limit_train_batches", 1.0),
        limit_val_batches=run_config["trainer"].get("limit_val_batches", 1.0),
        limit_test_batches=run_config["trainer"].get("limit_test_batches", 1.0),
        val_check_interval=run_config["trainer"].get("val_check_interval", 1.0),
        num_sanity_val_steps=run_config["trainer"].get("fast_dev_run", 3),

        # default log dir if no logger is found
        default_root_dir="logs/lightning_logs",
    )

    # Evaluate model on test set before training
    # trainer.test(model=lit_model, datamodule=datamodule)

    # Train the model
    trainer.fit(model=lit_model, datamodule=datamodule)

    # Evaluate model on test set after training
    trainer.test()


def load_config(path):
    with open(path, "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return config


def main(run_config_name: str, use_wandb: bool):
    # Load configs
    project_config: dict = load_config("project_config.yaml")
    run_config: dict = load_config("run_configs.yaml")[run_config_name]

    # Train model
    train(project_config=project_config, run_config=run_config, use_wandb=use_wandb)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--conf_name", type=str, default="MNIST_CLASSIFIER_V1")
    parser.add_argument("-u", "--use_wandb", type=bool, default=True)
    args = parser.parse_args()

    main(run_config_name=args.conf_name, use_wandb=args.use_wandb)
