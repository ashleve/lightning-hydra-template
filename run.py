import dotenv
import hydra
from omegaconf import DictConfig


# load environment variables from `.env` file
dotenv.load_dotenv(dotenv_path=".env", override=True)


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
    
    # nest imports inside method with @hydra.main to optimize tab completion
    from src.train import train
    
    return train(config)


if __name__ == "__main__":
    main()
