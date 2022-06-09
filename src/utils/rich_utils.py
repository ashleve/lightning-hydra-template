from pathlib import Path
from typing import Sequence

import rich
import rich.syntax
import rich.tree
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from rich.prompt import Prompt

from src.utils import get_pylogger

log = get_pylogger(__name__)


@rank_zero_only
def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = (
        "datamodule",
        "model",
        "callbacks",
        "logger",
        "trainer",
        "paths",
    ),
    resolve: bool = False,
    save_cfg_tree: bool = False,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
        print_order (Sequence[str], optional): Determines in what order config components are printed.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
        save_cfg_tree (bool, optional): Whether to export config to the hydra output folder.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # add fields from `print_order` to queue
    for field in print_order:
        queue.append(field) if field in cfg else log.warning(
            f"Field '{field}' not found in config. Skipping '{field}' config printing..."
        )

    # add all the other fields to queue (not specified in `print_order`)
    for field in cfg:
        if field not in queue:
            queue.append(field)

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # print config tree
    rich.print(tree)

    # save config tree to file
    if save_cfg_tree:
        with open(Path(cfg.paths.output_dir, "config_tree.log"), "w") as file:
            rich.print(tree, file=file)


def enforce_tags(cfg: DictConfig) -> None:
    """Prompts user to input tags from command line, if not tags are provided in config."""
    
    if not cfg.get("tags"):
        assert "id" not in HydraConfig().cfg.hydra.job, "Specify tags before launching multi-run!"

        tags = Prompt.ask("Enter a list of comma separated tags", default="dev")
        tags = [t.strip() for t in tags.split(",")]
        cfg.tags = tags

    log.info(f"Tags: {tags if tags is not None else []}")


if __name__ == "__main__":
    from hydra import compose, initialize

    with initialize(version_base="1.2", config_path="../../configs"):
        cfg = compose(config_name="train.yaml", return_hydra_config=True, overrides=[])
        print_config_tree(cfg, resolve=False, save_cfg_tree=False)
