import dotenv
import hydra
from omegaconf import DictConfig

from src.train import train
from src.utils import template_utils

# load environment variables from `.env` file if it exists
dotenv.load_dotenv(override=True)


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):

    # Imports can be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    # This is commented because DDP alongside Hydra doesn't work when imports are nested.
    # from src.train import train
    # from src.utils import template_utils

    # A couple of optional utilities:
    # - disabling python warnings
    # - disabling lightning logs
    # - easier access to debug mode
    # - forcing debug friendly configuration
    # You can safely get rid of this line if you don't want those
    template_utils.extras(config)

    # Pretty print config using Rich library
    if config.get("print_config"):
        template_utils.print_config(config, resolve=True)

    # Train model
    return train(config)


if __name__ == "__main__":
    main()
