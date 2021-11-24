import hydra
from copy import deepcopy
from typing import List
import string
import random
import uuid

import pytorch_lightning
from pytorch_lightning import Trainer, LightningModule, Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import LoggerCollection, LightningLoggerBase

import wandb
import numpy as np

from src.utils import utils

log = utils.get_logger(__name__)


class CV:
    """Cross-validation with a LightningModule."""
    def __init__(self,
                 config):
        super().__init__()
        self.config = config

    @staticmethod
    def update_modelcheckpoint(model_ckpt_callback, fold_idx):
        _default_filename = '{epoch}-{step}'
        _suffix = f'_fold{fold_idx}'
        if model_ckpt_callback.filename is None:
            new_filename = _default_filename + _suffix
        else:
            new_filename = model_ckpt_callback.filename + _suffix
        setattr(model_ckpt_callback, 'filename', new_filename)

    @staticmethod
    def init_logger(config, name: str, fold_idx: int):
        logger: List[LightningLoggerBase] = []
        if "logger" in config:
            for _, lg_conf in config.logger.items():
                if "_target_" in lg_conf:
                    log.info(f"Instantiating logger <{lg_conf._target_}>")
                    logger.append(hydra.utils.instantiate(lg_conf, name=name + f'_fold_{fold_idx + 1}'))

        return logger

    @staticmethod
    def init_callback(config):
        callbacks: List[Callback] = []
        if "callbacks" in config:
            for _, cb_conf in config.callbacks.items():
                if "_target_" in cb_conf:
                    log.info(f"Instantiating callback <{cb_conf._target_}>")
                    callbacks.append(hydra.utils.instantiate(cb_conf))
        return callbacks

    @staticmethod
    def logger_name_generator(string_length=10):
        random = str(uuid.uuid4())
        random = random.upper()
        random = random.replace("-", "")
        return random[0:string_length]

    def fit(self, model: LightningModule, data):
        splits = data.get_splits()
        scores = []

        if self.config.name is None:
            name = self.logger_name_generator()
        else:
            name = self.config.name
        for fold_idx, loaders in enumerate(splits):
            # Clone model
            _model = deepcopy(model)

            # init a new trainer
            callbacks = self.init_callback(self.config)
            logger = self.init_logger(self.config, name, fold_idx)
            trainer: Trainer = hydra.utils.instantiate(
                self.config.trainer, callbacks=callbacks, logger=logger, _convert_="partial")

            for callback in trainer.callbacks:
                if isinstance(callback, ModelCheckpoint):
                    self.update_modelcheckpoint(callback, fold_idx)

            # Fit:
            trainer.fit(_model, *loaders)

            wandb.finish()

            # Get metric score for hyperparameter optimization
            scores.append(trainer.callback_metrics.get(self.config.get("optimized_metric")))

        avg_score = round(100 * np.mean(scores), 2)
        return avg_score


