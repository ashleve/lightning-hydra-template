import logging
from functools import wraps
from typing import Callable, Optional, ParamSpec, TypeVar

from lightning_utilities.core.rank_zero import rank_prefixed_message, rank_zero_only


def get_ranked_pylogger(name: str = __name__) -> logging.Logger:
    """Initializes a multi-GPU-friendly python command line logger that logs on all processes with
    their rank prefixed in the log message.

    :param name: The name of the logger, defaults to ``__name__``.

    :return: A logger object.
    """
    T = TypeVar("T")
    P = ParamSpec("P")

    def _rank_prefixed_log(fn: Callable[P, T]) -> Callable[P, Optional[T]]:
        """Wrap a logging function to prefix its message with the rank of the process it's being
        logged from.

        If `'rank'` is provided in the wrapped functions kwargs, then the log will only occur on
        that rank/process.
        """

        @wraps(fn)
        def wrapped_fn(*args: P.args, **kwargs: P.kwargs) -> Optional[T]:
            rank = getattr(rank_zero_only, "rank", None)
            if rank is None:
                raise RuntimeError("The `rank_zero_only.rank` needs to be set before use")
            rank_to_log = kwargs.get("rank", None)
            msg = rank_prefixed_message(args[0], rank)
            if rank_to_log is None:
                return fn(msg=msg, *args[1:], **kwargs)
            elif rank == rank_to_log:
                return fn(msg=msg, *args[1:], **kwargs)
            else:
                return None

        return wrapped_fn

    logger = logging.getLogger(name)

    # This ensures all logging levels get marked with the _rank_prefixed_log decorator.
    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    for level in logging_levels:
        setattr(logger, level, _rank_prefixed_log(getattr(logger, level)))

    return logger
