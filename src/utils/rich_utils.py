import rich.syntax
import rich.tree
from typing import Sequence
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from src import utils
from pathlib import Path

log = utils.get_pylogger(__name__)


@rank_zero_only
def print_config(
    cfg: DictConfig,
    print_order: Sequence[str] = (
        "datamodule",
        "model",
        "callbacks",
        "logger",
        "trainer",
        "paths",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
        print_order (Sequence[str], optional): Determines in what order config components are printed.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # fetch config parts specified in `print_order`
    for field in print_order:
        queue.append(field) if field in cfg else log.info(
            f"Field '{field}' not found in config. Skipping '{field}' config printing..."
        )

    # if some config parts are not specified in `print_order`, print them at the end 
    for field in cfg:
        if field not in queue:
            queue.append(field)

    # generate config tree for printing
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    # save config tree to file
    with open(Path(cfg.paths.output_dir, "config_tree.log"), "w") as file:
        rich.print(tree, file=file)


# TODO rich logger?
