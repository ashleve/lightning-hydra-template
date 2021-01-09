from sklearn.metrics import accuracy_score
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
from template_utils.initializers import init_optimizer

# import custom architectures
from pytorch_modules.architectures.simple_mnist import SimpleMNISTClassifier


class LitModel(pl.LightningModule):
    """
    This is example of lightning model for MNIST classification.

    The path to model should be specified in your run config. (run_configs.yaml)
    The 'hparams' dict contains model args specified in config. (run_configs.yaml)

    This class enables you to specify what happens during training, validation and test step.
    You can just remove 'validation_step()' or 'test_step()' methods if you don't want to have them during training.

    To learn how to create lightning models visit:
        https://pytorch-lightning.readthedocs.io/en/latest/lightning_module.html
    """

    def __init__(self, optimizer_config, **hparams):
        super().__init__()
        self.optimizer_config = optimizer_config
        self.save_hyperparameters(hparams)

        self.model = SimpleMNISTClassifier(hparams=hparams)

    def forward(self, x):
        return self.model(x)

    # logic for a single training step
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.nll_loss(logits, y)

        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy_score(preds.cpu(), y.cpu())
        self.log('train_loss', loss, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, logger=True, prog_bar=True)

        # we can return here anything and then read it in some callback
        return loss, acc, preds, y

    # logic for a single validation step
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.nll_loss(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy_score(preds.cpu(), y.cpu())
        self.log('val_loss', loss, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, logger=True, prog_bar=True)

        return loss, acc, preds, y

    # logic for a single testing step
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.nll_loss(logits, y)

        # test metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy_score(preds.cpu(), y.cpu())
        self.log('test_loss', loss, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True, logger=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return init_optimizer(optimizer_config=self.optimizer_config, model=self)
