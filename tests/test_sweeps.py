import pytest

from tests.helpers.run_if import RunIf
from tests.helpers.run_sh_command import run_sh_command

startfile = "src/train.py"
overrides = ["++trainer.fast_dev_run=true", "logger=[]"]


@RunIf(sh=True)
@pytest.mark.slow
def test_experiments(tmp_path):
    """Test running all available experiment configs with fast_dev_run."""
    command = [
        startfile,
        "-m",
        "experiment=glob(*)",
        "hydra.sweep.dir=" + str(tmp_path),
    ] + overrides
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_default_sweep(tmp_path):
    """Test default Hydra sweeper."""
    command = [
        startfile,
        "-m",
        "hydra.sweep.dir=" + str(tmp_path),
        "model.lr=0.01,0.02,0.03",
        "trainer=default",
    ] + overrides

    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_optuna_sweep(tmp_path):
    """Test Optuna sweeper."""
    command = [
        startfile,
        "-m",
        "hparams_search=mnist_optuna",
        "hydra.sweep.dir=" + str(tmp_path),
        "hydra.sweeper.n_trials=10",
        "hydra.sweeper.sampler.n_startup_trials=5",
    ] + overrides
    run_sh_command(command)
