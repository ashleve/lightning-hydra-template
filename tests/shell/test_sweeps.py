import pytest

from tests.helpers.run_sh_command import run_sh_command
from tests.helpers.run_if import RunIf

"""
A couple of tests executing hydra sweeps.

Use the following command to skip slow tests:
    pytest -k "not slow"
"""


startfile = "train.py"
overrides = []


@RunIf(sh=True)
@pytest.mark.slow
def test_experiments():
    """Test running all available experiment configs with fast_dev_run."""
    command = [
        startfile,
        "-m",
        "experiment=glob(*)",
        "++trainer.fast_dev_run=true",
    ] + overrides
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_default_sweep():
    """Test default Hydra sweeper."""
    command = [
        startfile,
        "-m",
        "datamodule.batch_size=64,128",
        "model.lr=0.01,0.02",
        "trainer=default",
        "++trainer.fast_dev_run=true",
    ] + overrides
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_optuna_sweep():
    """Test Optuna sweeper."""
    command = [
        startfile,
        "-m",
        "hparams_search=mnist_optuna",
        "trainer=default",
        "++trainer.fast_dev_run=true",
    ] + overrides
    run_sh_command(command)
