from sklearn.metrics import accuracy_score
import pytorch_lightning as pl
import torch.nn.functional as F
import torch

# custom models
from models.simple_mnist_classifier.models import *


class LitModel(pl.LightningModule):

    def __init__(self, hparams=None):
        super().__init__()

        if hparams:
            self.save_hyperparameters(hparams)

        self.model = SimpleMNISTClassifier(config=self.hparams)

    def forward(self, x):
        return self.model(x)

    # logic for a single training step
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.nll_loss(logits, y)

        # training metrics
        preds = torch.argmax(logits, dim=1)
        preds, y = preds.cpu(), y.cpu()
        acc = accuracy_score(preds, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, logger=True, prog_bar=True)

        return loss

    # logic for a single validation step
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.nll_loss(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        preds, y = preds.cpu(), y.cpu()
        acc = accuracy_score(preds, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, logger=True, prog_bar=True)

        # we can return here anything and then read it in some callback
        return preds, y

    # logic for a single testing step
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.nll_loss(logits, y)

        # test metrics
        preds = torch.argmax(logits, dim=1)
        preds, y = preds.cpu(), y.cpu()
        acc = accuracy_score(preds, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True, logger=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
