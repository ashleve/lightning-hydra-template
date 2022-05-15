import hydra
from hydra import initialize, compose
from train import main
from omegaconf import OmegaConf
from rich import print
import os
from pathlib import Path


def test_train_fast_dev_run(tmpdir):
    with initialize(config_path="../../configs/"):
        cfg = compose(config_name="train", overrides=["++trainer.fast_dev_run=true"])
        cfg.original_work_dir = os.getcwd()

        main(cfg)


# assert cfg == {
#     "app": {"user": "test_user", "num1": 10, "num2": 20},
#     "db": {"host": "localhost", "port": 3306},
# }

# assert False
