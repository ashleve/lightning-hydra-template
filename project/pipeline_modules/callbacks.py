from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pytorch_lightning.callbacks import Callback
import seaborn
import torch
import wandb
import os


class ExampleCallback(Callback):
    def __init__(self):
        pass

    def on_init_start(self, trainer):
        print('Starting to initialize trainer!')

    def on_init_end(self, trainer):
        print('Trainer is initialized now.')

    def on_train_end(self, trainer, pl_module):
        print('Do something when training ends.')


class SaveModelOnnxCallback(Callback):
    """
        Save model in .onnx format.
    """
    def __init__(self, datamodule, save_dir):
        first_batch = next(iter(datamodule.train_dataloader()))
        x, y = first_batch
        self.dummy_input = x
        self.save_dir = save_dir

    def on_sanity_check_end(self, trainer, pl_module):
        self.save_onnx_model(pl_module)

    def on_train_end(self, trainer, pl_module):
        self.save_onnx_model(pl_module)

    def save_onnx_model(self, pl_module):
        file_path = os.path.join(self.save_dir, "model.onnx")
        pl_module.to_onnx(file_path=file_path, input_sample=self.dummy_input.to(pl_module.device))
        wandb.save(file_path)


class ImagePredictionLoggerCallback(Callback):
    """
        Each epoch upload to wandb a couple of the same images with predicted labels.
    """
    def __init__(self, datamodule, num_samples=8):
        first_batch = next(iter(datamodule.train_dataloader()))
        self.imgs, self.labels = first_batch
        self.imgs, self.labels = self.imgs[:num_samples], self.labels[:num_samples]
        self.ready = True

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.ready:
            imgs = self.imgs.to(device=pl_module.device)
            logits = pl_module(imgs)
            preds = torch.argmax(logits, -1)
            trainer.logger.experiment.log({f"img_examples": [
                wandb.Image(
                    x,
                    caption=f"Epoch: {trainer.current_epoch} Pred:{pred}, Label:{y}"
                ) for x, pred, y in zip(imgs, preds, self.labels)
            ]}, commit=False)


class UnfreezeModelCallback(Callback):
    """
        Unfreeze model after a few epochs.
    """
    def __init__(self, wait_epochs=5):
        self.wait_epochs = wait_epochs

    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch == self.wait_epochs:
            for param in pl_module.model.model.parameters():
                param.requires_grad = True


class MetricsHeatmapLoggerCallback(Callback):
    """
        Generate f1, precision and recall heatmap calculated from each validation epoch outputs.
        Expects validation step to return predictions and targets.
    """
    def __init__(self):
        self.preds = []
        self.targets = []
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Gather data from single batch."""
        if self.ready:
            preds, targets = outputs
            self.preds.append(preds)
            self.targets.append(targets)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate f1, precision and recall heatmap."""
        if self.ready:
            self.preds = torch.cat(self.preds)
            self.targets = torch.cat(self.targets)
            f1 = f1_score(self.preds, self.targets, average=None)
            r = recall_score(self.preds, self.targets, average=None)
            p = precision_score(self.preds, self.targets, average=None)

            trainer.logger.experiment.log({
                f"f1_p_r_heatmap_{trainer.current_epoch}": wandb.plots.HeatMap(
                    x_labels=[str(i) for i in range(len(f1))],  # class names can be hardcoded here
                    y_labels=["f1", "precision", "recall"],
                    matrix_values=[f1, p, r],
                    show_text=True,
                )}, commit=False)

            self.preds = []
            self.targets = []
