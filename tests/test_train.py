import os

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from src.tasks import train_task
from tests.helpers.run_if import RunIf


def run(cfg):
    HydraConfig().set_config(cfg)
    train_task.train(cfg)


def test_train_fast_dev_run(cfg_train):
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
    run(cfg_train)


@RunIf(min_gpus=1)
def test_train_fast_dev_run_gpu(cfg_train):
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.gpus = 1
    run(cfg_train)


@RunIf(min_gpus=1)
def test_train_fast_dev_run_gpu_amp(cfg_train):
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.gpus = 1
        cfg_train.trainer.precision = 16
    run(cfg_train)


@pytest.mark.slow
def test_train_epoch(cfg_train):
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
    run(cfg_train)


@pytest.mark.slow
def test_train_epoch_double_val_loop(cfg_train):
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.trainer.val_check_interval = 0.5
    run(cfg_train)


@pytest.mark.slow
def test_train_resume(tmp_path, cfg_train):
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1

    run(cfg_train)

    files = os.listdir(tmp_path / "checkpoints")
    assert "last.ckpt" in files
    assert "epoch_000.ckpt" in files

    with open_dict(cfg_train):
        cfg_train.ckpt_path = str(tmp_path / "checkpoints" / "last.ckpt")
        cfg_train.trainer.max_epochs = 2

    run(cfg_train)

    files = os.listdir(tmp_path / "checkpoints")
    assert "epoch_001.ckpt" in files
    assert "epoch_002.ckpt" not in files
