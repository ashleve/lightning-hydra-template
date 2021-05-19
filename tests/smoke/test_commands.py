import pytest

from tests.helpers.run_command import run_command
from tests.helpers.runif import RunIf


def test_fast_dev_run():
    """Run 1 train, val, test batch."""
    command = ["run.py", "trainer=default", "trainer.fast_dev_run=true"]
    run_command(command)


def test_default_cpu():
    """Test default configuration on CPU."""
    command = ["run.py", "trainer.max_epochs=1", "trainer.gpus=0"]
    run_command(command)


@RunIf(min_gpus=1)
def test_default_gpu():
    """Test default configuration on GPU."""
    command = [
        "run.py",
        "trainer.max_epochs=1",
        "trainer.gpus=1",
        "datamodule.pin_memory=True",
    ]
    run_command(command)


@pytest.mark.slow
def test_experiments():
    """Train 1 epoch with all experiment configs."""
    command = ["run.py", "-m", "experiment=glob(*)", "trainer.max_epochs=1"]
    run_command(command)


def test_limit_batches():
    """Train 1 epoch on 25% of data."""
    command = [
        "run.py",
        "trainer=default",
        "trainer.max_epochs=1",
        "trainer.limit_train_batches=0.25",
        "trainer.limit_val_batches=0.25",
        "trainer.limit_test_batches=0.25",
    ]
    run_command(command)


def test_gradient_accumulation():
    """Train 1 epoch with gradient accumulation."""
    command = [
        "run.py",
        "trainer=default",
        "trainer.max_epochs=1",
        "trainer.accumulate_grad_batches=10",
    ]
    run_command(command)


def test_double_validation_loop():
    """Train 1 epoch with validation loop twice per epoch."""
    command = [
        "run.py",
        "trainer=default",
        "trainer.max_epochs=1",
        "trainer.val_check_interval=0.5",
    ]
    run_command(command)


def test_csv_logger():
    """Train 5 epochs with 5 batches with CSVLogger."""
    command = [
        "run.py",
        "trainer=default",
        "trainer.max_epochs=5",
        "trainer.limit_train_batches=5",
        "logger=csv",
    ]
    run_command(command)


def test_tensorboard_logger():
    """Train 5 epochs with 5 batches with TensorboardLogger."""
    command = [
        "run.py",
        "trainer=default",
        "trainer.max_epochs=5",
        "trainer.limit_train_batches=5",
        "logger=tensorboard",
    ]
    run_command(command)


def test_overfit_batches():
    """Overfit to 10 batches over 10 epochs."""
    command = [
        "run.py",
        "trainer=default",
        "trainer.min_epochs=10",
        "trainer.max_epochs=10",
        "trainer.overfit_batches=10",
    ]
    run_command(command)
