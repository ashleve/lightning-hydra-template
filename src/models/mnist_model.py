from pytorch_lightning.metrics.classification import Accuracy
import pytorch_lightning as pl
import torch.nn.functional as F
import torch

# import custom architectures
from src.architectures.simple_dense_net import SimpleDenseNet


class LitModelMNIST(pl.LightningModule):
    """
    This is example of LightningModule for MNIST classification.
        https://pytorch-lightning.readthedocs.io/en/latest/lightning_module.html
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.accuracy = Accuracy()
        self.architecture = SimpleDenseNet(hparams=self.hparams)

        self.train_acc_hist = []
        self.train_loss_hist = []
        self.val_acc_hist = []
        self.val_loss_hist = []

    def forward(self, x):
        return self.architecture(x)

    # logic for a single training step
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y)

        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here anything and then read it in some callback or in training_epoch_end() below
        return {"loss": loss, "preds": preds, "targets": y}

    # logic for a single validation step
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here anything and then read it in some callback or in validation_epoch_end() below
        return {"loss": loss, "preds": preds, "targets": y}

    # logic for a single testing step
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y)

        # test metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return loss

    def training_epoch_end(self, outputs):
        self.train_acc_hist.append(self.trainer.callback_metrics["train/acc"])
        self.train_loss_hist.append(self.trainer.callback_metrics["train/loss"])
        self.log("train/acc_best", max(self.train_acc_hist), prog_bar=False)
        self.log("train/loss_best", min(self.train_loss_hist), prog_bar=False)

    def validation_epoch_end(self, outputs):
        self.val_acc_hist.append(self.trainer.callback_metrics["val/acc"])
        self.val_loss_hist.append(self.trainer.callback_metrics["val/loss"])
        self.log("val/acc_best", max(self.val_acc_hist), prog_bar=False)
        self.log("val/loss_best", min(self.val_loss_hist), prog_bar=False)

    def configure_optimizers(self):
        if self.hparams.optimizer == "adam":
            return torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            raise Exception("Invalid optimizer name")
