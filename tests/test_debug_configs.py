import pytest

from tests.helpers.run_sh_command import run_sh_command
from tests.helpers.run_if import RunIf

"""
A couple of sanity checks to make sure debugging configs don't crash.
"""


startfile = "train.py"
overrides = ["callbacks=none", "++trainer.max_steps=1", "logger=[]"]
# overrides = ["++trainer.max_steps=1", "logger=[]"]


# @RunIf(sh=True)
# @pytest.mark.slow
# def test_debug_default():
#     command = [startfile, "debug=default"] + overrides
#     run_sh_command(command)


# @RunIf(sh=True)
# def test_debug_limit_batches():
#     command = [startfile, "debug=limit_batches"] + overrides
#     run_sh_command(command)


# @RunIf(sh=True)
# def test_debug_overfit():
#     command = [startfile, "debug=overfit"] + overrides
#     run_sh_command(command)


# @RunIf(sh=True)
# @pytest.mark.slow
# def test_debug_profiler():
#     command = [startfile, "debug=profiler"] + overrides
#     run_sh_command(command)


# @RunIf(sh=True)
# def test_debug_step():
#     command = [startfile, "debug=step"] + overrides
#     run_sh_command(command)


# @RunIf(sh=True)
# def test_debug_test_only():
#     command = [startfile, "debug=test_only"] + overrides
#     run_sh_command(command)
