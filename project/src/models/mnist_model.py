from pytorch_lightning.metrics.classification import Accuracy
from src.utils.initializers import load_class
import pytorch_lightning as pl
import torch.nn.functional as F
import torch

# import custom architectures
from src.architectures.simple_dense_net import SimpleDenseNet


class LitModelMNIST(pl.LightningModule):
    """
    This is example of lightning model for MNIST classification.
    To learn how to create lightning models visit:
        https://pytorch-lightning.readthedocs.io/en/latest/lightning_module.html
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.accuracy = Accuracy()

        # Initialize model architecture
        if self.hparams["architecture"] == "SimpleDenseNet":
            self.architecture = SimpleDenseNet(hparams=self.hparams)
        else:
            raise Exception("Invalid architecture name")

    def forward(self, x):
        return self.architecture(x)

    # logic for a single training step
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.architecture(x)
        loss = F.nll_loss(logits, y)

        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    # logic for a single validation step
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.architecture(x)
        loss = F.nll_loss(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here anything and then read it in some callback
        return {"batch_val_loss": loss, "batch_val_acc": acc, "batch_val_preds": preds, "batch_val_y": y}

    # logic for a single testing step
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.architecture(x)
        loss = F.nll_loss(logits, y)

        # test metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        Optimizer = load_class(self.hparams["optimizer_config"]["class"])
        return Optimizer(self.parameters(), **self.hparams["optimizer_config"]["args"])

