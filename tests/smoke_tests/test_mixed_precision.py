from tests.helpers.run_command import run_command
from tests.helpers.runif import RunIf


@RunIf(amp_apex=True)
def test_apex_01():
    """Test mixed-precision level 01."""
    command = [
        "run.py",
        "trainer=default",
        "trainer.max_epochs=1",
        "trainer.gpus=1",
        "trainer.amp_backend=apex",
        "trainer.amp_level=O1",
        "trainer.precision=16",
    ]
    run_command(command)


@RunIf(amp_apex=True)
def test_apex_02():
    """Test mixed-precision level 02."""
    command = [
        "run.py",
        "trainer=default",
        "trainer.max_epochs=1",
        "trainer.gpus=1",
        "trainer.amp_backend=apex",
        "trainer.amp_level=O2",
        "trainer.precision=16",
    ]
    run_command(command)


@RunIf(amp_apex=True)
def test_apex_03():
    """Test mixed-precision level 03."""
    command = [
        "run.py",
        "trainer=default",
        "trainer.max_epochs=1",
        "trainer.gpus=1",
        "trainer.amp_backend=apex",
        "trainer.amp_level=O3",
        "trainer.precision=16",
    ]
    run_command(command)
