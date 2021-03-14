import hydra
from omegaconf import DictConfig


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
    
    # nest imports inside method with @hydra.main to optimize tab completion
    from src.train import train
    
    return train(config)


if __name__ == "__main__":
    main()
