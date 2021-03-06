import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.metrics.classification import Accuracy

# import custom architectures
from src.architectures.rnn_model import RNN


class LitModeIMDB(pl.LightningModule):
    """
    This is example of LightningModule for IMDB text classification.
        Based on https://github.com/bentrevett/pytorch-sentiment-analysis
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.accuracy = Accuracy()
        self.architecture = RNN(**self.hparams)

    def forward(self, text, text_lengths):
        return self.architecture(text, text_lengths)

    def step(self, batch, stage='train'):
        text, text_lengths = batch.text
        logits = self.forward(text, text_lengths).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, batch.label)

        # metrics
        preds = torch.round(torch.sigmoid(logits))
        acc = self.accuracy(preds, batch.label.int())
        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"{stage}/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    # logic for a single training step
    def training_step(self, batch, batch_idx):
        return self.step(batch, 'train')

    # logic for a single validation step
    def validation_step(self, batch, batch_idx):
        return self.step(batch, 'val')

    # logic for a single testing step
    def test_step(self, batch, batch_idx):
        return self.step(batch, 'test')

    def configure_optimizers(self):
        if self.hparams.optimizer == "adam":
            return torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            raise Exception("Invalid optimizer name")
