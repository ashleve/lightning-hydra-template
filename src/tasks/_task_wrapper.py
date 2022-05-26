import time

from omegaconf import DictConfig
from pytorch_lightning.utilities import rank_zero_only

from src import utils

log = utils.get_logger(__name__)


def task_wrapper(task_func):
    """Optional decorator that wraps the task function in extra utilities:

    - logging the total time of execution
    - enabling repeating task execution on failure
    - calling the utils.extras() before the task is started
    - calling the utils.finish() after the task is finished
    """

    def wrap(cfg: DictConfig):
        start = time.time()

        # applies optional config utilities
        utils.extras(cfg)

        # TODO: repeat call if fails...
        result = task_func(cfg=cfg)
        metric_value, object_dict = result

        # make sure everything closed properly
        utils.finish(object_dict)

        end = time.time()
        save_exec_time(task_name=task_func.__name__, time_in_seconds=end - start)

        assert isinstance(metric_value, float) or metric_value is None
        assert isinstance(object_dict, dict)
        return metric_value, object_dict

    return wrap


@rank_zero_only
def save_exec_time(task_name, time_in_seconds):
    with open("execution_time.log", "w+") as file:
        file.write("Total execution time.\n")
        file.write(task_name + ": " + str(time_in_seconds) + "\n")
