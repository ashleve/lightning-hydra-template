from typing import Callable

from omegaconf import DictConfig

from src.utils import pylogger
from src.utils.utils import close_loggers

log = pylogger.get_pylogger(__name__)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    By the template design, the "task" is a function which returns a tuple: (metric_dict, object_dict).

    This wrapper can be used to:
    - make sure that the loggers are closed even if the task function raises an exception (prevents multirun failure)
    - save the exception to a `.log` file
    - mark the run as failed before raising exception, with a dedicated file in the `logs/` folder (so we can find and rerun it later)
    - etc. (depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[dict, dict]:

        ...

        return metric_dict, object_dict
    ```
    """

    def wrap(cfg: DictConfig):

        # execute the task
        try:

            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:

            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # when using hyperparameter search plugins like Optuna you might want to disable raising exception
            # to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:

            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # close loggers (even if exception occurs so multirun won't fail)
            close_loggers()

        return metric_dict, object_dict

    return wrap
