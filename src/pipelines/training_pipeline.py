from typing import Any, Dict, List, Optional, Tuple

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase

from src import utils
from src.pipelines import pipeline_wrapper

log = utils.get_logger(__name__)


@pipeline_wrapper
def train(cfg: DictConfig) -> Tuple[Optional[float], Dict[str, Any]]:
    """Contains the training pipeline.

    Can additionally evaluate model on a testset, using best
    weights obtained during training.

    This method is wrapped in @pipeline_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[Optional[float], Dict[str, Any]]: A tuple of metric value for hyperparameter optimization
        and dictionary with all instantiated objects.
    """

    # init lightning datamodule
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    # init lightning model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # init lightning callbacks
    log.info("Instantiating callbacks!")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    # init lightning loggers
    log.info("Instantiating loggers!")
    logger: List[LightningLoggerBase] = utils.instantiate_loggers(cfg.get("logger"))

    # init lightning trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    # send hyperparameters to loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(object_dict)

    # train the model
    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    # get metric value for hyperparameter optimization
    metric_value = None
    if cfg.get("optimized_metric"):
        log.info("Retrieving metric value!")
        metric_value = utils.get_metric_value(metric_name=cfg.optimized_metric, trainer=trainer)

    # test the model
    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = "best" if trainer.checkpoint_callback.best_model_path else None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    # print path to best checkpoint
    if trainer.checkpoint_callback.best_model_path:
        log.info(f"Best model ckpt at {trainer.checkpoint_callback.best_model_path}")

    # return metric value for hyperparameter optimization
    return metric_value, object_dict
