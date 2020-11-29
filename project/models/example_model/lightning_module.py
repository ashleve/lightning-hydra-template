import pytorch_lightning as pl
import torch.nn.functional as F
import torch

# custom models
from models.example_model.models import *


class LitModel(pl.LightningModule):
    """
        This is example of lightning model.
        It should always be located in file named 'lightninig_module.py' and always be named 'LitModel'!

        The folder name of 'LitModel' used during training should be specified in run config and all parameters from
        'model' section will be passed in 'hparams' dictionary.

        It enables you to specify what happens during training, validation and test step.
        You can just remove 'validation_step()' or 'test_step()' methods if you don't want to have them during training.

        See 'simple_mnist_classifier' for more proper example.
    """

    def __init__(self, hparams=None):
        super().__init__()

        if hparams:
            self.save_hyperparameters(hparams)

        self.model = ExampleModel(hparams=self.hparams)

    def forward(self, x):
        return self.model(x)

    # logic for a single training step
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.nll_loss(logits, y)

        return loss

    # logic for a single validation step
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.nll_loss(logits, y)

        return loss

    # logic for a single testing step
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.nll_loss(logits, y)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
