import hydra
import pyrootutils
from omegaconf import DictConfig

# project root setup (allows for calling this script from any place)
# searches for root indicator in parent directories, like ".git", "pyproject.toml", "setup.py", etc.
# sets PROJECT_ROOT environment variable (used for specifying paths in hydra config)
# loads environment variables from ".env" file if exists
# adds project root directory to the PYTHONPATH (so imports work correctly)
# https://github.com/ashleve/pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    project_root_env_var=True,
    dotenv=True,
    pythonpath=True,
    cwd=False,  # do NOT change working directory to root (would cause problems in DDP mode)
)


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="train.yaml")
def main(cfg: DictConfig) -> float:

    # imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src.tasks.train_task import train

    # train model
    metric_value, _ = train(cfg)

    # return metric value for hyperparameter optimization
    return metric_value


if __name__ == "__main__":
    main()
