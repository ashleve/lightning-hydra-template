from pytorch_lightning import Callback


class ExampleCallback(Callback):
    def __init__(self):
        pass

    def on_init_start(self, trainer):
        print("Starting to initialize trainer!")

    def on_init_end(self, trainer):
        print("Trainer is initialized now.")

    def on_train_end(self, trainer, pl_module):
        print("Do something when training ends.")


class UnfreezeModelCallback(Callback):
    """
    Unfreeze all model parameters after a few epochs.
    """

    def __init__(self, wait_epochs=5):
        self.wait_epochs = wait_epochs

    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch == self.wait_epochs:
            for param in pl_module.model.model.parameters():
                param.requires_grad = True
