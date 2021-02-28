# pytorch lightning imports
from pytorch_lightning import LightningModule, LightningDataModule, Callback, Trainer
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning import seed_everything

# hydra imports
from omegaconf import DictConfig
import hydra

# normal imports
from typing import List

# src imports
from src.utils import template_utils as utils


def train(config: DictConfig):

    # Pretty print config using Rich library
    if config["print_config"]:
        utils.print_config(config)

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config["seed"])

    # Init PyTorch Lightning model ⚡
    model: LightningModule = hydra.utils.instantiate(config["model"])

    # Init PyTorch Lightning datamodule ⚡
    datamodule: LightningDataModule = hydra.utils.instantiate(config["datamodule"])
    datamodule.prepare_data()
    datamodule.setup()

    # Init PyTorch Lightning callbacks ⚡
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, callback_conf in config["callbacks"].items():
            if "_target_" in callback_conf:
                callbacks.append(hydra.utils.instantiate(callback_conf))

    # Init PyTorch Lightning loggers ⚡
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, logger_conf in config["logger"].items():
            if "_target_" in logger_conf:
                logger.append(hydra.utils.instantiate(logger_conf))

    # Init PyTorch Lightning trainer ⚡
    trainer: Trainer = hydra.utils.instantiate(
        config["trainer"], callbacks=callbacks, logger=logger
    )

    # Send some hyperparameters to all lightning loggers
    utils.log_hparams_to_all_loggers(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Train the model
    trainer.fit(model=model, datamodule=datamodule)

    # Evaluate model on test set after training
    trainer.test()

    # Make sure everything closed properly.
    # utils.finish(
    #     config=config,
    #     model=model,
    #     datamodule=datamodule,
    #     trainer=trainer,
    #     callbacks=callbacks,
    #     logger=logger,
    # )

    # Return best achieved metric score for optuna
    optimized_metric = config.get("optimized_metric", None)
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
    return train(config)


if __name__ == "__main__":
    main()
