import pytest
import train

from tests.helpers.run_if import RunIf
from omegaconf import open_dict
from tests.helpers import load_config


def test_train_fast_dev_run(tmp_path):
    cfg = load_config.load_train_cfg_simple(tmp_path)
    train.main(cfg)


@RunIf(min_gpus=1)
def test_train_fast_dev_run_gpu(tmp_path):
    cfg = load_config.load_train_cfg_simple(tmp_path)
    with open_dict(cfg):
        cfg.trainer.gpus = 1
    train.main(cfg)


@RunIf(min_gpus=1)
def test_train_fast_dev_run_gpu_amp(tmp_path):
    cfg = load_config.load_train_cfg_simple(tmp_path)
    with open_dict(cfg):
        cfg.trainer.gpus = 1
        cfg.trainer.precision = 16
    train.main(cfg)


@pytest.mark.slow
def test_train_epoch(tmp_path):
    cfg = load_config.load_train_cfg_simple(tmp_path)
    with open_dict(cfg):
        cfg.trainer.fast_dev_run = False
        cfg.trainer.max_epochs = 1
        cfg.trainer.gpus = 0
    train.main(cfg)


@pytest.mark.slow
def test_train_epoch_double_val_loop(tmp_path):
    cfg = load_config.load_train_cfg_simple(tmp_path)
    with open_dict(cfg):
        cfg.trainer.fast_dev_run = False
        cfg.trainer.max_epochs = 1
        cfg.trainer.gpus = 0
    train.main(cfg)
