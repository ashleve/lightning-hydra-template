import platform
from importlib.util import find_spec

"""
Adapted from:
    https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/utilities/imports.py
"""


def _module_available(module_path: str) -> bool:
    """Check if a path is available in your environment.

    >>> _module_available('os')
    True
    >>> _module_available('bla.bla')
    False
    """
    try:
        return find_spec(module_path) is not None
    except ModuleNotFoundError:
        # Python 3.7+
        return False


_IS_WINDOWS = platform.system() == "Windows"
_DEEPSPEED_AVAILABLE = not _IS_WINDOWS and _module_available("deepspeed")
_FAIRSCALE_AVAILABLE = not _IS_WINDOWS and _module_available("fairscale.nn")
_RPC_AVAILABLE = not _IS_WINDOWS and _module_available("torch.distributed.rpc")
