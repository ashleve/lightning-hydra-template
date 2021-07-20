import pytest

from tests.helpers.run_command import run_command
from tests.helpers.runif import RunIf


def test_fast_dev_run():
    """Run 1 train, val, test batch."""
    command = ["run.py", "++trainer.fast_dev_run=true"]
    run_command(command)


def test_default_cpu():
    """Test default config on CPU."""
    command = ["run.py", "++trainer.max_epochs=1", "++trainer.gpus=0"]
    run_command(command)


# use RunIf to skip execution of some tests, e.g. when no gpus are available
@RunIf(min_gpus=1)
def test_default_gpu():
    """Test default config on GPU."""
    command = [
        "run.py",
        "++trainer.max_epochs=1",
        "++trainer.gpus=1",
        "++datamodule.pin_memory=True",
    ]
    run_command(command)


@pytest.mark.slow
def test_experiments():
    """Train 1 epoch with all experiment configs."""
    command = ["run.py", "-m", "experiment=glob(*)", "++trainer.max_epochs=1"]
    run_command(command)


def test_limit_batches():
    """Train 1 epoch on 25% of data."""
    command = [
        "run.py",
        "++trainer.max_epochs=1",
        "++trainer.limit_train_batches=0.25",
        "++trainer.limit_val_batches=0.25",
        "++trainer.limit_test_batches=0.25",
    ]
    run_command(command)


def test_double_validation_loop():
    """Train 1 epoch with validation loop twice per epoch."""
    command = [
        "run.py",
        "++trainer.max_epochs=1",
        "++trainer.val_check_interval=0.5",
    ]
    run_command(command)
