# pytorch
import torch
import torch.nn.functional as F

# pytorch lightning
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy, precision, recall, f1_score, fbeta_score

# custom models
from pipeline_modules.models import ModelMNISTv1, ModelMNISTv2


class LitModel(pl.LightningModule):

    def __init__(self, config):
        super().__init__()

        self.save_hyperparameters(config["hparams"])
        self.model = ModelMNISTv2(config=self.hparams)

    def forward(self, x):
        return self.model(x)

    # logic for a single training step
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.nll_loss(logits, y)

        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log('train_loss', loss, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_epoch=True, logger=True)

        return loss

    # logic for a single validation step
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.nll_loss(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.log('val_acc', acc, prog_bar=True, logger=True)

        return loss

    # logic for a single testing step
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.nll_loss(logits, y)

        # test metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log('test_loss', loss, prog_bar=True, logger=True)
        self.log('test_acc', acc, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
