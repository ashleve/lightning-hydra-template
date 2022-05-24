import os

import pytest
from omegaconf import open_dict

from src import test, train
from tests.helpers import load_config


@pytest.mark.slow
def test_train_resume(tmp_path):
    cfg = load_config.load_train_cfg_simple(tmp_path)

    with open_dict(cfg):
        cfg.trainer.fast_dev_run = False
        cfg.trainer.max_epochs = 1

    train.main(cfg)

    files = os.listdir(tmp_path)
    assert "last.ckpt" in files
    assert "epoch_000.ckpt" in files

    with open_dict(cfg):
        cfg.ckpt_path = str(tmp_path / "last.ckpt")
        cfg.trainer.max_epochs = 2

    train.main(cfg)

    files = os.listdir(tmp_path)
    assert "epoch_001.ckpt" in files


@pytest.mark.slow
def test_train_eval(tmp_path):
    cfg_train = load_config.load_train_cfg_simple(tmp_path)

    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = False
        cfg_train.trainer.max_epochs = 1

    train.main(cfg_train)

    files = os.listdir(tmp_path)
    assert "last.ckpt" in files

    cfg_test = load_config.load_test_cfg_simple(tmp_path, ckpt_path=str(tmp_path / "last.ckpt"))
    test.main(cfg_test)
