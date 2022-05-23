import pyrootutils
import hydra
from omegaconf import DictConfig


root = pyrootutils.setup_root(indicator=".git", search_from=__file__)


@hydra.main(config_path="configs", config_name="test.yaml")
def main(cfg: DictConfig):

    from src.pipelines.testing_pipeline import test

    test(cfg)


if __name__ == "__main__":
    main()
