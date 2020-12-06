from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.loggers import WandbLogger
from argparse import ArgumentParser
import pytorch_lightning as pl
from typing import List
import yaml
import os

# utils
from utils.init_utils import init_lit_model, init_data_module, init_main_callbacks, init_wandb_logger
from utils.callbacks import *


def train(project_config: dict, run_config: dict, use_wandb: bool):
    # Init PyTorch Lightning model ⚡
    lit_model: pl.LightningModule = init_lit_model(hparams=run_config["model"])

    # Init PyTorch Lightning datamodule ⚡
    datamodule: pl.LightningDataModule = init_data_module(hparams=run_config["dataset"])

    # Init Weights&Biases logger
    logger: pl.loggers.WandbLogger = init_wandb_logger(
        project_config=project_config,
        run_config=run_config,
        lit_model=lit_model,
        datamodule=datamodule,
        log_path=os.path.join(os.path.dirname(__file__), "logs/")
    ) if use_wandb else None

    # Init ModelCheckpoint and EarlyStopping callbacks
    callbacks: List[pl.Callback] = init_main_callbacks(project_config=project_config)

    # Add custom callbacks from utils/callbacks.py
    callbacks.extend([
        # MetricsHeatmapLoggerCallback(),
        # UnfreezeModelCallback(wait_epochs=5),
    ])
    if use_wandb:
        callbacks.append(
            SaveCodeToWandbCallback(
                base_dir=os.path.dirname(__file__),
                wandb_save_dir=logger.save_dir,
                run_config=run_config
            ),
        )

    # Get path to checkpoint you want to resume with if it was set in the run config
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
        if resume_from_checkpoint != "None" and resume_from_checkpoint is not False
        else None,

        # print related
        progress_bar_refresh_rate=project_config["printing"]["progress_bar_refresh_rate"],
        profiler=SimpleProfiler() if project_config["printing"]["profiler"] else None,
        weights_summary=project_config["printing"]["weights_summary"],

        # number of validation sanity checks
        num_sanity_val_steps=3,

        # default log dir if no logger is found
        default_root_dir=os.path.dirname(__file__) + "logs/lightning_logs",

        # insert all other trainer parameters specified in run config
        **run_config["trainer"]
    )

    # Evaluate model on test set before training
    # trainer.test(model=lit_model, datamodule=datamodule)

    # Train the model
    trainer.fit(model=lit_model, datamodule=datamodule)

    # Evaluate model on test set after training
    trainer.test()


def load_config(filename):
    path = os.path.join(os.path.dirname(__file__), filename)
    with open(path, "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return config


def main(run_config_name: str, use_wandb: bool):
    # Load configs
    project_config: dict = load_config("project_config.yaml")
    run_config: dict = load_config("run_configs.yaml")[run_config_name]

    # Print run config info
    print("EXECUTING RUN:", run_config_name)
    print("CONFIG:")
    for section in run_config:
        print("  " + section + ":")
        for key in run_config[section]:
            print("    " + key + ":", run_config[section][key])

    # Train model
    train(project_config=project_config, run_config=run_config, use_wandb=use_wandb)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-r", "--run_config", type=str, default="MNIST_CLASSIFIER_V1")
    parser.add_argument("-u", "--use_wandb", type=bool, default=True)
    args = parser.parse_args()

    main(run_config_name=args.run_config, use_wandb=args.use_wandb)
