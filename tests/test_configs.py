import os
import hydra
from omegaconf import OmegaConf
from pathlib import Path
from omegaconf import DictConfig


def test_train_config(cfg_train: DictConfig):
    assert cfg_train
    assert cfg_train.datamodule
    assert cfg_train.model
    assert cfg_train.trainer

    hydra.utils.instantiate(cfg_train.datamodule)
    hydra.utils.instantiate(cfg_train.model)
    hydra.utils.instantiate(cfg_train.trainer)


def test_test_config(cfg_test: DictConfig):
    assert cfg_test
    assert cfg_test.datamodule
    assert cfg_test.model
    assert cfg_test.trainer

    hydra.utils.instantiate(cfg_test.datamodule)
    hydra.utils.instantiate(cfg_test.model)
    hydra.utils.instantiate(cfg_test.trainer)
