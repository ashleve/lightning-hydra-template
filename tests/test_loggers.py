from tests.helpers.run_if import RunIf


@RunIf(wandb=True)
def test_wandb(cfg_train):
    pass


def test_csv(cfg_train):
    pass
