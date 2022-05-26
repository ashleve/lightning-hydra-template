import hydra
import pyrootutils
from omegaconf import DictConfig

# project root setup
# allows for calling this script from any place
root = pyrootutils.setup_root(
    indicator=".git",  # file which indicates the root dir
    search_from=__file__,  # search for indicator in parent dirs, starting from here
    pythonpath=True,  # add root dir to the PYTHONPATH
    cwd=True,  # set current work dir to the root dir
    dotenv=True,  # load environment variables from .env if exists
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
