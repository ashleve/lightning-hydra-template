import os
import pytest
from omegaconf import DictConfig
import train
from hydra import initialize, compose
from hydra.core.hydra_config import HydraConfig


def test_train_resume(cfg_train: DictConfig):
    cfg_train.trainer.max_epochs = 1

    train.main(cfg_train)
    
    
def test_train_eval(cfg_train: DictConfig):
    cfg_train.trainer.max_epochs = 1

    train.main(cfg_train)


# def test_train_resume():
#     with initialize(config_path="../../configs/"):
#         cfg = compose(config_name="train", overrides=["++trainer.fast_dev_run=true"])
#         cfg.original_work_dir = os.getcwd()
#         HydraConfig().set_config(cfg)
#         train.main(cfg)
