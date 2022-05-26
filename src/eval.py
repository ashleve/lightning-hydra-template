import hydra
import pyrootutils
from omegaconf import DictConfig

root = pyrootutils.setup_root(__file__)


@hydra.main(config_path=root / "configs", config_name="eval.yaml")
def main(cfg: DictConfig):

    from src.tasks.eval_task import eval

    eval(cfg)


if __name__ == "__main__":
    main()
