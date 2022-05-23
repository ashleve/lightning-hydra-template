import hydra
import pyrootutils
from omegaconf import DictConfig

root = pyrootutils.setup_root(
    indicator=".git",
    search_from=__file__,
    pythonpath=True,
    cwd=True,
    dotenv=True,
)


@hydra.main(config_path=root / "configs", config_name="test.yaml")
def main(cfg: DictConfig):

    from src.pipelines.testing_pipeline import test

    test(cfg)


if __name__ == "__main__":
    main()
