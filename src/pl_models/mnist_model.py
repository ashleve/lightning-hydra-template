from typing import Any, Dict, List, Sequence, Tuple, Union

import hydra
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.metrics.classification import Accuracy
from torch.optim import Optimizer

from src.architectures.simple_dense_net import SimpleDenseNet


class MNISTLitModel(LightningModule):
    """
    Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/lightning_module.html
    """

    def __init__(
        self,
        optimizer,
        input_size=784,
        lin1_size=256,
        lin2_size=256,
        lin3_size=256,
        output_size=10,
        **kwargs
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.model = SimpleDenseNet(hparams=self.hparams)

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

        self.metric_hist = {
            "train/acc": [],
            "val/acc": [],
            "train/loss": [],
            "val/loss": [],
        }

    def forward(self, x) -> torch.Tensor:
        return self.model(x)

    def step(self, batch) -> Dict[str, torch.Tensor]:
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        loss, preds, targets = self.step(batch)

        # log train metrics
        acc = self.train_accuracy(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc = self.val_accuracy(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_accuracy(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return loss

    # [OPTIONAL METHOD]
    def training_epoch_end(self, outputs: List[Any]) -> None:
        # log best so far train acc and train loss
        self.metric_hist["train/acc"].append(self.trainer.callback_metrics["train/acc"])
        self.metric_hist["train/loss"].append(
            self.trainer.callback_metrics["train/loss"]
        )
        self.log("train/acc_best", max(self.metric_hist["train/acc"]), prog_bar=False)
        self.log("train/loss_best", min(self.metric_hist["train/loss"]), prog_bar=False)

    # [OPTIONAL METHOD]
    def validation_epoch_end(self, outputs: List[Any]) -> None:
        # log best so far val acc and val loss
        self.metric_hist["val/acc"].append(self.trainer.callback_metrics["val/acc"])
        self.metric_hist["val/loss"].append(self.trainer.callback_metrics["val/loss"])
        self.log("val/acc_best", max(self.metric_hist["val/acc"]), prog_bar=False)
        self.log("val/loss_best", min(self.metric_hist["val/loss"]), prog_bar=False)

    # [OPTIONAL METHOD]
    def test_epoch_end(self, outputs: List[Any]) -> None:
        pass

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers

        """
        optim = hydra.utils.instantiate(
            self.hparams.optimizer, params=self.parameters()
        )
        return optim
