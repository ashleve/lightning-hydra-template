from tests.helpers.run_if import RunIf


@RunIf(wandb=True)
def test_wandb(tmp_path):
    pass
