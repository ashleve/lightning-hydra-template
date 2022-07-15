import hydra
import pyrootutils
from omegaconf import DictConfig

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:

    from src.tasks.eval_task import evaluate

    evaluate(cfg)


if __name__ == "__main__":
    main()
