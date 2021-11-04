from tests.helpers.run_command import run_command
from tests.helpers.runif import RunIf

"""
A couple of sanity checks to make sure the model doesn't crash with different running options.
"""


def test_fast_dev_run():
    """Test running for 1 train, val and test batch."""
    command = ["run.py", "++trainer.fast_dev_run=true"]
    run_command(command)


def test_cpu():
    """Test running 1 epoch on CPU."""
    command = ["run.py", "++trainer.max_epochs=1", "++trainer.gpus=0"]
    run_command(command)


# use RunIf to skip execution of some tests, e.g. when no gpus are available
@RunIf(min_gpus=1)
def test_gpu():
    """Test running 1 epoch on GPU."""
    command = [
        "run.py",
        "++trainer.max_epochs=1",
        "++trainer.gpus=1",
    ]
    run_command(command)


@RunIf(min_gpus=1)
def test_mixed_precision():
    """Test running 1 epoch with pytorch native automatic mixed precision (AMP)."""
    command = [
        "run.py",
        "++trainer.max_epochs=1",
        "++trainer.gpus=1",
        "++trainer.precision=16",
    ]
    run_command(command)


def test_limit_batches():
    """Test running 1 epoch on 25% of data."""
    command = [
        "run.py",
        "++trainer.max_epochs=1",
        "++trainer.limit_train_batches=0.25",
        "++trainer.limit_val_batches=0.25",
        "++trainer.limit_test_batches=0.25",
    ]
    run_command(command)


def test_double_validation_loop():
    """Test running 1 epoch with validation loop twice per epoch."""
    command = [
        "run.py",
        "++trainer.max_epochs=1",
        "++trainer.val_check_interval=0.5",
    ]
    run_command(command)
