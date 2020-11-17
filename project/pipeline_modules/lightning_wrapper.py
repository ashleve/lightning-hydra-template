from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pl_bolts.models.regression import LogisticRegression
from pl_bolts.models.regression import LinearRegression
from torch.nn.functional import one_hot
import torch.nn.functional as F
import pytorch_lightning as pl
import torch
import wandb

# custom models
from pipeline_modules.models import *


class LitModel(pl.LightningModule):

    def __init__(self, config):
        super().__init__()

        self.save_hyperparameters(config["hparams"])
        self.model = SimpleLinearMNIST(config=self.hparams)

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
        # f1 = f1_score(preds, y, average="micro")
        # self.log('train_f1', f1, on_step=False, on_epoch=True, logger=True, prog_bar=True)

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
        # f1 = f1_score(preds, y, average="micro")
        # self.log('val_f1', f1, on_step=False, on_epoch=True, logger=True, prog_bar=True)

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
        # f1 = f1_score(preds, y, average="micro")
        # self.log('test_f1', f1, on_step=False, on_epoch=True, logger=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams["weight_decay"])
