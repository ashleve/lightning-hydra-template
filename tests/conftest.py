import os
import pytest

from pytest import TempPathFactory
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra import compose, initialize


def load_cfg(config_name: str, tmp_dir: str) -> DictConfig:
    with initialize(config_path="../configs"):
        cfg = compose(config_name=config_name, return_hydra_config=True)

        # set work dir to pytest temporary dir
        cfg.hydra.run.dir = os.path.join(tmp_dir, cfg.hydra.run.dir)
        cfg.hydra.sweep.dir = os.path.join(tmp_dir, cfg.hydra.sweep.dir)

        # enable adding new keys to config
        OmegaConf.set_struct(cfg, True)

        # set default config options for all tests
        with open_dict(cfg):
            cfg.print_config = False
            cfg.ignore_warnings = False
            cfg.train = True
            cfg.test = True
            cfg.datamodule.num_workers = 0
            cfg.datamodule.pin_memory = False
            cfg.trainer.fast_dev_run = True

        return cfg


@pytest.fixture(scope="package")
def cfg_train(tmp_path_factory: TempPathFactory) -> DictConfig:
    tmpdir_train = tmp_path_factory.mktemp("tmpdir_train")
    cfg = load_cfg("train.yaml", tmpdir_train)
    yield cfg


@pytest.fixture(scope="package")
def cfg_test(tmp_path_factory: TempPathFactory) -> DictConfig:
    tmpdir_test = tmp_path_factory.mktemp("tmpdir_test")
    cfg = load_cfg("test.yaml", tmpdir_test)
    yield cfg
