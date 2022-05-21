import time
from src import utils
from omegaconf import DictConfig

log = utils.get_logger(__name__)


def pipeline_wrapper(pipeline_func):
    """Optional decorator that wraps the pipeline function in extra utilities:

    - logging the total time of execution
    - enabling repeating pipeline call on failure
    - calling the utils.extras() before the pipeline is started
    - calling the utils.finish() after the pipeline is finished
    """

    def wrap(cfg: DictConfig):
        start = time.time()

        # applies optional config utilities
        utils.extras(cfg)

        # TODO: repeat call if fails...
        result = pipeline_func(cfg=cfg)
        metric_value, object_dict = result

        # make sure everything closed properly
        utils.finish(object_dict)

        end = time.time()

        with open("execution_time.log", "w+") as file:
            file.write("Total execution time:\n")
            file.write(pipeline_func.__name__ + ": " + str(end - start) + "\n")

        assert type(metric_value) is float or metric_value is None
        assert type(object_dict) is dict
        return metric_value, object_dict

    return wrap
