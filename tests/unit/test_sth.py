import pytest

from tests.helpers.runif import RunIf


def test_something1():
    """Some test description."""
    assert True is True


def test_something2():
    """Some test description."""
    assert 1 + 1 == 2


@pytest.mark.parametrize("arg1", [0.5, 1.0, 2.0])
def test_something3(arg1: float):
    """Some test description."""
    assert arg1 > 0


# use RunIf to skip execution of some tests, e.g. when not on windows or when no gpus are available
@RunIf(skip_windows=True, min_gpus=1)
def test_something4():
    """Some test description."""
    assert True is True
