from omegaconf import DictConfig, OmegaConf, open_dict
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from typing import List


def load_cfg(config_name: str, overrides: List[str]) -> DictConfig:
    with initialize(config_path="../../configs"):
        cfg = compose(config_name=config_name, return_hydra_config=True, overrides=overrides)

        # enable adding new keys to config
        OmegaConf.set_struct(cfg, True)

        # enable config interpolation
        HydraConfig().set_config(cfg)

        return cfg


def load_train_cfg_simple(tmp_dir: str) -> DictConfig:
    cfg = load_cfg("train.yaml", overrides=[])

    # set defaults
    with open_dict(cfg):
        cfg.trainer.default_root_dir = str(tmp_dir)

        if cfg.get("callbacks") and cfg.callbacks.get("model_checkpoint"):
            cfg.callbacks.model_checkpoint.dirpath = str(tmp_dir)

        cfg.print_config = False
        cfg.ignore_warnings = False
        cfg.datamodule.num_workers = 0
        cfg.datamodule.pin_memory = False
        cfg.trainer.fast_dev_run = True
        cfg.trainer.gpus = 0

    return cfg


def load_test_cfg_simple(tmp_dir: str, ckpt_path: str) -> DictConfig:
    cfg = load_cfg("test.yaml", overrides=[f"ckpt_path={ckpt_path}"])

    # set defaults
    with open_dict(cfg):
        cfg.trainer.default_root_dir = str(tmp_dir)
        cfg.print_config = False
        cfg.ignore_warnings = False
        cfg.datamodule.num_workers = 0
        cfg.datamodule.pin_memory = False
        cfg.trainer.fast_dev_run = True
        cfg.trainer.gpus = 0

    return cfg
