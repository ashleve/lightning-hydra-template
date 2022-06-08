import hydra
import pyrootutils
from omegaconf import DictConfig

# project root setup
# searches for root indicators in parent dirs, like ".git", "pyproject.toml", "setup.py", etc.
# https://github.com/ashleve/pyrootutils
pyrootutils.setup_root(
    search_from=__file__,
    project_root_env_var=True,  # set PROJECT_ROOT env var (used in `configs/paths/default.yaml`)
    dotenv=True,  # load env vars from ".env" if exists
    pythonpath=True,  # add root dir to the PYTHONPATH
)


@hydra.main(version_base="1.2", config_path="../configs", config_name="train.yaml")
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
