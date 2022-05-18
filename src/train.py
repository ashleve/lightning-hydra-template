import hydra
from omegaconf import DictConfig
from typing import Optional


@hydra.main(config_path="../configs/", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:

    # imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from pipelines.training_pipeline import train

    # train model
    score, _ = train(cfg)

    # return metric score for hyperparameter optimization
    return score


if __name__ == "__main__":
    main()
