from tests.helpers.run_command import run_command


def test_default_sweep():
    """Test default Hydra sweeper."""
    command = [
        "run.py",
        "-m",
        "datamodule.batch_size=64,128",
        "model.lr=0.01,0.02",
        "trainer=default",
        "trainer.fast_dev_run=true",
    ]
    run_command(command)


def test_optuna_sweep():
    """Test Optuna sweeper."""
    command = [
        "run.py",
        "-m",
        "hparams_search=mnist_optuna",
        "trainer=default",
        "trainer.fast_dev_run=true",
    ]
    run_command(command)


# def test_ax_sweep():
#     """Test Ax sweeper."""
#     command = ["run.py", "-m", "hparams_search=mnist_ax", "trainer.fast_dev_run=true"]
#     run_command(command)
