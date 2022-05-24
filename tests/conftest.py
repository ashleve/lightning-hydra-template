"""Use the following command to skip slow tests:

pytest -k "not slow"
"""

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from pytest import TempPathFactory

from tests.helpers import load_config


@pytest.fixture(scope="package")
def cfg_train(tmp_path_factory: TempPathFactory) -> DictConfig:
    tmpdir_train = tmp_path_factory.mktemp("tmpdir_train")
    cfg = load_config.load_train_cfg_simple(tmpdir_train)
    yield cfg


@pytest.fixture(scope="package")
def cfg_test(tmp_path_factory: TempPathFactory) -> DictConfig:
    tmpdir_test = tmp_path_factory.mktemp("tmpdir_test")
    cfg = load_config.load_eval_cfg_simple(tmpdir_test, ckpt_path=str(tmpdir_test / "last.ckpt"))
    yield cfg
