from utils import run_command


def test_wandb_optuna_sweep():
    """Test wandb logging with Optuna sweep."""
    command = [
        "run.py",
        "-m",
        "hparams_search=mnist_optuna",
        "trainer=default",
        "trainer.max_epochs=10",
        "trainer.limit_train_batches=20",
        "logger=wandb",
        "logger.wandb.project=template-tests",
        "logger.wandb.group=Optuna_SimpleDenseNet_MNIST",
        "hydra.sweeper.n_trials=5",
    ]
    run_command(command)


def test_wandb_callbacks():
    """Test wandb callbacks."""
    command = [
        "run.py",
        "trainer=default",
        "trainer.max_epochs=3",
        "logger=wandb",
        "logger.wandb.project=template-tests",
        "callbacks=wandb",
    ]
    run_command(command)
