import pytest

from tests.helpers.runif import RunIf


def test_something():
    """Some test description."""
    assert True is True
    

# use RunIf to skip execution of some tests, e.g. when not on windows or when no gpus are available
@RunIf(skip_windows=True, min_gpus=1)
def test_something_else():
    """Some test description."""
    assert True is True


@pytest.mark.parametrize("arg1", [0.5, 1.0])
def test_even_more(arg1: float):
    """Some test description."""
    assert arg1 > 0
