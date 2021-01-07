import torch
import yaml
from omegaconf import DictConfig, OmegaConf
import hydra.conf.hydra.output
from hydra.experimental import initialize, compose
import comet_ml
from pytorch_lightning.loggers import CSVLogger, WandbLogger, CometLogger, TestTubeLogger, TensorBoardLogger
import logging


log = logging.getLogger(__name__)


@hydra.main(config_path="configs/", config_name="config.yaml")
def my_app(cfg: DictConfig) -> None:
    # pass
    # print(cfg)
    log.info(OmegaConf.to_yaml(cfg, resolve=True))


def test_logger():
    logger = WandbLogger()
    print(logger)


if __name__ == "__main__":
    my_app()

    # with open("epoch=0.ckpt", "rb") as file:
    #     data = torch.load(file)
    #     print(data)

    # test_logger()
