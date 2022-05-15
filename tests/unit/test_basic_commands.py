import os
import pytest

from hydra import initialize, compose
import train

from tests.helpers.run_sh_command import run_sh_command
from tests.helpers.run_if import RunIf

"""
A couple of sanity checks to make sure basic commands work.
"""


startfile = "train.py"


def test_train_fast_dev_run(tmpdir):
    with initialize(config_path="../../configs/"):
        cfg = compose(config_name="train", overrides=["++trainer.fast_dev_run=true"])
        cfg.original_work_dir = os.getcwd()
        train.main(cfg)


@RunIf(sh=True)
def test_fast_dev_run():
    """Test running for 1 train, val and test batch."""
    command = [startfile, "++trainer.fast_dev_run=true"]
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_cpu():
    """Test running 1 epoch on CPU."""
    command = [startfile, "++trainer.max_epochs=1", "++trainer.gpus=0"]
    run_sh_command(command)


# use RunIf to skip execution of some tests, e.g. when no gpus are available
@RunIf(sh=True, min_gpus=1)
@pytest.mark.slow
def test_gpu():
    """Test running 1 epoch on GPU."""
    command = [
        startfile,
        "++trainer.max_epochs=1",
        "++trainer.gpus=1",
    ]
    run_sh_command(command)


@RunIf(sh=True, min_gpus=1)
@pytest.mark.slow
def test_mixed_precision():
    """Test running 1 epoch with pytorch native automatic mixed precision (AMP)."""
    command = [
        startfile,
        "++trainer.max_epochs=1",
        "++trainer.gpus=1",
        "++trainer.precision=16",
    ]
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_double_validation_loop():
    """Test running 1 epoch with validation loop twice per epoch."""
    command = [
        startfile,
        "++trainer.max_epochs=1",
        "++trainer.val_check_interval=0.5",
    ]
    run_sh_command(command)
