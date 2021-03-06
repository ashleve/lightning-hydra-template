# lightning imports
from pytorch_lightning import LightningModule, LightningDataModule, Callback, Trainer
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning import seed_everything

# hydra imports
from omegaconf import DictConfig
from hydra.utils import log
import hydra

# normal imports
from typing import List
import warnings
import logging

# src imports
from src.utils import template_utils as utils


def train(config: DictConfig):

    # Disable python warnings
    if config.disable_warnings:
        log.info(
            f"Disabling python warnings! <disable_warnings={config.disable_warnings}>"
        )
        warnings.filterwarnings("ignore")

    # Disable Lightning logs
    if config.disable_lightning_logs:
        log.info(
            f"Disabling lightning logs! <disable_lightning_logs={config.disable_lightning_logs}>"
        )
        logging.getLogger("lightning").setLevel(logging.ERROR)

    # Force debugger friendly configuration
    if config.trainer.fast_dev_run:
        log.info(
            f"Forcing debugger friendly configuration! <fast_dev_run={config.trainer.fast_dev_run}>"
        )
        config = utils.convert_config_to_debug_fiendly(config)

    # Pretty print config using Rich library
    if config.print_config:
        log.info(
            f"Pretty printing config with Rich! " "<print_config={config.print_config}>"
        )
        utils.print_config(config)

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        log.info(f"Setting seed! <seed={config.seed}>")
        seed_everything(config.seed)

    # Init Lightning model ⚡
    log.info(f"Initializing LightningModule! <_target_={config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)

    # Init Lightning datamodule ⚡
    log.info(
        f"Initializing LightningDataModule! <_target_={config.datamodule._target_}>"
    )
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    datamodule.prepare_data()
    datamodule.setup()

    # Init Lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Initializing Callback! <_target_={cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init Lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config["logger"].items():
            if "_target_" in lg_conf:
                log.info(f"Initializing Logger! <_target_={lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init Lightning trainer ⚡
    log.info(f"Initializing Trainer! <_target_={config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config["trainer"], callbacks=callbacks, logger=logger
    )

    # Send some parameters from config to all lightning loggers
    log.info(f"Logging hyperparameters!")
    utils.log_hparams_to_all_loggers(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Train the model ⚡
    log.info(f"Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    # Evaluate model on test set after training
    log.info(f"Testing!")
    trainer.test()

    # Make sure everything closed properly
    log.info(f"Finalizing!")
    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Return metric score for optuna optimization
    optimized_metric = config.get("optimized_metric", None)
    if optimized_metric:
        result = trainer.callback_metrics[optimized_metric]
        log.info(f"Returning metric for Optuna! <{optimized_metric}={result}>")
        return result


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
    return train(config)


if __name__ == "__main__":
    main()
