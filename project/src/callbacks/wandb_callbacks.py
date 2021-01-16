from sklearn.metrics import precision_score, recall_score, f1_score
from pytorch_lightning.loggers import WandbLogger
from wandb.sdk.wandb_run import Run as wandb_run
from pytorch_lightning import Callback
import pytorch_lightning as pl
import torch
import wandb
import glob
import os


def get_wandb_logger(trainer: pl.Trainer) -> wandb_run:
    logger = None
    for some_logger in trainer.logger.experiment:
        if isinstance(some_logger, wandb_run):
            logger = some_logger

    if not logger:
        raise Exception("You're using wandb related callback, "
                        "but wandb logger was not initialized for some reason...")

    return logger


class SaveCodeToWandb(Callback):
    """
    Upload all *.py files to wandb as an artifact at the beginning of the run.
    """
    def __init__(self, code_dir: str):
        self.code_dir = code_dir

    def on_sanity_check_end(self, trainer, pl_module):
        """Upload files when all validation sanity checks end."""
        logger = get_wandb_logger(trainer=trainer)

        code = wandb.Artifact('project-source', type='code')
        for path in glob.glob(os.path.join(self.code_dir, '**/*.py'), recursive=True):
            code.add_file(path)
        wandb.run.use_artifact(code)


class UploadAllCheckpointsToWandb(Callback):
    """
    Upload experiment checkpoints to wandb as an artifact at the end of training.
    """
    def __init__(self, ckpt_dir: str = "checkpoints/", upload_best_only: bool = False):
        self.ckpt_dir = ckpt_dir
        self.upload_best_only = upload_best_only

    def on_train_end(self, trainer, pl_module):
        """Upload ckpts when training ends."""
        logger = get_wandb_logger(trainer=trainer)

        ckpts = wandb.Artifact('experiment-ckpts', type='checkpoints')
        if self.upload_best_only:
            ckpts.add_file(trainer.checkpoint_callback.best_model_path)
        else:
            for path in glob.glob(os.path.join(self.ckpt_dir, '**/*.ckpt'), recursive=True):
                ckpts.add_file(path)
        wandb.run.use_artifact(ckpts)


class SaveMetricsHeatmapToWandb(Callback):
    """
    Generate f1, precision and recall heatmap from validation step outputs.
    Expects validation step to return predictions and targets.
    Works only for single label classification!
    """
    def __init__(self, class_names=None):
        self.class_names = class_names
        self.preds = []
        self.targets = []
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Gather data from single batch."""
        if self.ready:
            preds, targets = outputs["batch_val_preds"], outputs["batch_val_y"]
            self.preds.append(preds)
            self.targets.append(targets)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate f1, precision and recall heatmap."""
        if self.ready:
            logger = get_wandb_logger(trainer=trainer)

            self.preds = torch.cat(self.preds).cpu()
            self.targets = torch.cat(self.targets).cpu()
            f1 = f1_score(self.preds, self.targets, average=None)
            r = recall_score(self.preds, self.targets, average=None)
            p = precision_score(self.preds, self.targets, average=None)

            logger.log({
                f"f1_p_r_heatmap_{trainer.current_epoch}_{logger.id}": wandb.plots.HeatMap(
                    x_labels=self.class_names,
                    y_labels=["f1", "precision", "recall"],
                    matrix_values=[f1, p, r],
                    show_text=True,
                )}, commit=False)

            self.preds = []
            self.targets = []


class SaveConfusionMatrixToWandb(Callback):
    """
    Generate Confusion Matrix.
    Expects validation step to return predictions and targets.
    Works only for single label classification!
    """
    def __init__(self, class_names=None):
        self.class_names = class_names
        self.preds = []
        self.targets = []
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Gather data from single batch."""
        if self.ready:
            preds, targets = outputs["batch_val_preds"], outputs["batch_val_y"]
            self.preds.append(preds)
            self.targets.append(targets)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate f1, precision and recall heatmap."""
        if self.ready:
            logger = get_wandb_logger(trainer=trainer)

            self.preds = torch.cat(self.preds).tolist()
            self.targets = torch.cat(self.targets).tolist()

            logger.log({
                f"conf_mat_{trainer.current_epoch}_{logger.id}": wandb.plot.confusion_matrix(
                    preds=self.preds,
                    y_true=self.targets,
                    class_names=self.class_names)
            }, commit=False)

            self.preds = []
            self.targets = []


class SaveBestMetricScoresToWandb(Callback):
    """
    Store in wandb:
        - max train acc
        - min train loss
        - max val acc
        - min val loss
    Useful for comparing runs in table views, as wandb doesn't currently supports column aggregation.
    """
    def __init__(self):
        self.train_loss_best = None
        self.train_acc_best = None
        self.val_loss_best = None
        self.val_acc_best = None
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_epoch_end(self, trainer, pl_module):
        if self.ready:
            logger = get_wandb_logger(trainer=trainer)

            metrics = trainer.callback_metrics
            if self.train_loss_best is None or metrics["train_loss"] < self.train_loss_best:
                self.train_loss_best = metrics["train_loss"]
            if self.train_acc_best is None or metrics["train_acc"] > self.train_acc_best:
                self.train_acc_best = metrics["train_acc"]
            if self.val_loss_best is None or metrics["val_loss"] < self.val_loss_best:
                self.val_loss_best = metrics["val_loss"]
            if self.val_acc_best is None or metrics["val_acc"] > self.val_acc_best:
                self.val_acc_best = metrics["val_acc"]

            logger.log({"train_loss_best": self.train_loss_best}, commit=False)
            logger.log({"train_acc_best": self.train_acc_best}, commit=False)
            logger.log({"val_loss_best": self.val_loss_best}, commit=False)
            logger.log({"val_acc_best": self.val_acc_best}, commit=False)


# class SaveImagePredictionsToWandb(Callback):
#     """
#     Each epoch upload to wandb a couple of the same images with predicted labels.
#     """
#     def __init__(self, datamodule, num_samples=8):
#         first_batch = next(iter(datamodule.train_dataloader()))
#         self.imgs, self.labels = first_batch
#         self.imgs, self.labels = self.imgs[:num_samples], self.labels[:num_samples]
#         self.ready = True
#
#     def on_sanity_check_end(self, trainer, pl_module):
#         """Start executing this callback only after all validation sanity checks end."""
#         self.ready = True
#
#     def on_validation_epoch_end(self, trainer, pl_module):
#         if self.ready:
#             imgs = self.imgs.to(device=pl_module.device)
#             logits = pl_module(imgs)
#             preds = torch.argmax(logits, -1)
#             trainer.logger.experiment.log({f"img_examples": [
#                 wandb.Image(
#                     x,
#                     caption=f"Epoch: {trainer.current_epoch} Pred:{pred}, Label:{y}"
#                 ) for x, pred, y in zip(imgs, preds, self.labels)
#             ]}, commit=False)
