from typing import List, Dict, Tuple, Any


import hydra
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase

from src.pipelines import pipeline_wrapper
from src import utils

log = utils.get_logger(__name__)


@pipeline_wrapper
def test(cfg: DictConfig) -> Tuple[None, Dict[str, Any]]:
    """Contains minimal example of the testing pipeline. Evaluates given checkpoint on a testset.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[None, Dict[str, Any]]
    """

    assert cfg.ckpt_path

    # init lightning datamodule
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    # init lightning model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # init lightning loggers
    log.info("Instantiating loggers...")
    logger: List[LightningLoggerBase] = utils.instantiate_loggers(cfg.get("logger"))

    # init lightning trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    # create dict for more convenient access to objects
    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    # send hyperparameters to loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(object_dict)

    # test the model
    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    return None, object_dict
