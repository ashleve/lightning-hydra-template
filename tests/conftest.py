import os
import pytest

from pytest import TempPathFactory
from omegaconf import DictConfig
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig


def init_hydra_cfg(config_name: str, tmpdir: str) -> DictConfig:
    with initialize(config_path="../configs"):
        cfg = compose(config_name=config_name, return_hydra_config=True)

        cfg.hydra.run.dir = os.path.join(tmpdir, cfg.hydra.run.dir)
        cfg.hydra.sweep.dir = os.path.join(tmpdir, cfg.hydra.sweep.dir)

        HydraConfig().set_config(cfg)

        return cfg


@pytest.fixture(scope="package")
def cfg_train(tmp_path_factory: TempPathFactory) -> DictConfig:
    tmpdir_train = tmp_path_factory.mktemp("tmpdir_train")
    cfg = init_hydra_cfg("train", tmpdir_train)
    yield cfg


@pytest.fixture(scope="package")
def cfg_test(tmp_path_factory: TempPathFactory) -> DictConfig:
    tmpdir_test = tmp_path_factory.mktemp("tmpdir_test")
    cfg = init_hydra_cfg("test", tmpdir_test)
    yield cfg
