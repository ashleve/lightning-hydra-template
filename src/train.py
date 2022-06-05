import hydra
import pyrootutils
from omegaconf import DictConfig

# project root setup (allows for calling this script from any place)
# searches for root indicators in parent directories, like ".git", "pyproject.toml", "setup.py", etc.
# adds project root directory to the PYTHONPATH
# sets current working directory to the root directory
# loads environment variables from ".env" file if exists
# https://github.com/ashleve/pyrootutils
root = pyrootutils.setup_root(__file__)


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
