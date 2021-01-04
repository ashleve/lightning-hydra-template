from omegaconf import DictConfig, OmegaConf
import hydra.conf.hydra.output
from hydra.experimental import initialize, compose
import comet_ml


@hydra.main(config_path="configs/", config_name="project_config.yaml")
def my_app(cfg: DictConfig) -> None:
    # print(cfg)
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    my_app()
