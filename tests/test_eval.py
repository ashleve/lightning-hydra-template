import os

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from src.tasks import eval_task, train_task


@pytest.mark.slow
def test_train_eval(tmp_path, cfg_train, cfg_eval):
    assert str(tmp_path) == cfg_train.paths.output_dir == cfg_eval.paths.output_dir

    print(tmp_path)
    print(cfg_train.paths.output_dir)
    print(cfg_train.callbacks.model_checkpoint.dirpath)

    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1

    HydraConfig().set_config(cfg_train)
    train_task.train(cfg_train)

    print(os.listdir(tmp_path))

    assert "last.ckpt" in os.listdir(tmp_path / "checkpoints")

    with open_dict(cfg_eval):
        cfg_eval.ckpt_path = str(tmp_path / "checkpoints" / "last.ckpt")

    HydraConfig().set_config(cfg_eval)
    eval_task.evaluate(cfg_eval)
