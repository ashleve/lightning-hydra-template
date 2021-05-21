import sys
from typing import Optional

import pytest
import torch
from packaging.version import Version
from pkg_resources import get_distribution

"""
Adapted from:
    https://github.com/PyTorchLightning/pytorch-lightning/blob/master/tests/helpers/runif.py
"""

from tests.helpers.module_available import (
    _APEX_AVAILABLE,
    _DEEPSPEED_AVAILABLE,
    _FAIRSCALE_AVAILABLE,
    _IS_WINDOWS,
    _RPC_AVAILABLE,
)


class RunIf:
    """
    RunIf wrapper for conditional skipping of tests.
    Fully compatible with `@pytest.mark`.

    Example:

        @RunIf(min_torch="1.8")
        @pytest.mark.parametrize("arg1", [1.0, 2.0])
        def test_wrapper(arg1):
            assert arg1 > 0

    """

    def __new__(
        self,
        min_gpus: int = 0,
        min_torch: Optional[str] = None,
        max_torch: Optional[str] = None,
        min_python: Optional[str] = None,
        amp_apex: bool = False,
        skip_windows: bool = False,
        rpc: bool = False,
        fairscale: bool = False,
        deepspeed: bool = False,
        **kwargs,
    ):
        """
        Args:
            min_gpus: min number of gpus required to run test
            min_torch: minimum pytorch version to run test
            max_torch: maximum pytorch version to run test
            min_python: minimum python version required to run test
            amp_apex: NVIDIA Apex is installed
            skip_windows: skip test for Windows platform
            rpc: requires Remote Procedure Call (RPC)
            fairscale: if `fairscale` module is required to run the test
            deepspeed: if `deepspeed` module is required to run the test
            kwargs: native pytest.mark.skipif keyword arguments
        """
        conditions = []
        reasons = []

        if min_gpus:
            conditions.append(torch.cuda.device_count() < min_gpus)
            reasons.append(f"GPUs>={min_gpus}")

        if min_torch:
            torch_version = get_distribution("torch").version
            conditions.append(Version(torch_version) < Version(min_torch))
            reasons.append(f"torch>={min_torch}")

        if max_torch:
            torch_version = get_distribution("torch").version
            conditions.append(Version(torch_version) >= Version(max_torch))
            reasons.append(f"torch<{max_torch}")

        if min_python:
            py_version = (
                f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            )
            conditions.append(Version(py_version) < Version(min_python))
            reasons.append(f"python>={min_python}")

        if amp_apex:
            conditions.append(not _APEX_AVAILABLE)
            reasons.append("NVIDIA Apex")

        if skip_windows:
            conditions.append(_IS_WINDOWS)
            reasons.append("does not run on Windows")

        if rpc:
            conditions.append(not _RPC_AVAILABLE)
            reasons.append("RPC")

        if fairscale:
            conditions.append(not _FAIRSCALE_AVAILABLE)
            reasons.append("Fairscale")

        if deepspeed:
            conditions.append(not _DEEPSPEED_AVAILABLE)
            reasons.append("Deepspeed")

        reasons = [rs for cond, rs in zip(conditions, reasons) if cond]
        return pytest.mark.skipif(
            condition=any(conditions),
            reason=f"Requires: [{' + '.join(reasons)}]",
            **kwargs,
        )
