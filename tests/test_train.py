import pytest

import train

from tests.helpers.run_if import RunIf
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict


def test_train_fast_dev_run(cfg_train: DictConfig):
    HydraConfig().set_config(cfg_train)
    train.main(cfg_train)


@RunIf(min_gpus=1)
def test_train_fast_dev_run_gpu(cfg_train: DictConfig):
    with open_dict(cfg_train):
        cfg_train.trainer.gpus = 1
    HydraConfig().set_config(cfg_train)
    train.main(cfg_train)


@RunIf(min_gpus=1)
def test_train_fast_dev_run_gpu_amp(cfg_train: DictConfig):
    with open_dict(cfg_train):
        cfg_train.trainer.gpus = 1
        cfg_train.trainer.precision = 16
    HydraConfig().set_config(cfg_train)
    train.main(cfg_train)


@pytest.mark.slow
def test_train_epoch_cpu(cfg_train: DictConfig):
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = False
        cfg_train.trainer.max_epochs = 1
        cfg_train.trainer.gpus = 0
    HydraConfig().set_config(cfg_train)
    train.main(cfg_train)


@RunIf(min_gpus=1)
@pytest.mark.slow
def test_train_epoch_gpu(cfg_train: DictConfig):
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = False
        cfg_train.trainer.max_epochs = 1
        cfg_train.trainer.gpus = 1
    HydraConfig().set_config(cfg_train)
    train.main(cfg_train)


@pytest.mark.slow
def test_train_epoch_double_val_loop(cfg_train: DictConfig):
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = False
        cfg_train.trainer.max_epochs = 1
        cfg_train.trainer.val_check_interval = 0.5
    HydraConfig().set_config(cfg_train)
    train.main(cfg_train)
    