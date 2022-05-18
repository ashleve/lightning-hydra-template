import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs/", config_name="test.yaml")
def main(cfg: DictConfig):

    from src.pipelines.testing_pipeline import test

    # evaluate model
    test(cfg)


if __name__ == "__main__":
    main()
