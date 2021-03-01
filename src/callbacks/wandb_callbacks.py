# wandb
from pytorch_lightning.loggers import WandbLogger
import wandb

# pytorch
from pytorch_lightning import Callback
import pytorch_lightning as pl
import torch

# others
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import List
import glob
import os


def get_wandb_logger(trainer: pl.Trainer) -> WandbLogger:
    logger = None
    for lg in trainer.logger:
        if isinstance(lg, WandbLogger):
            logger = lg

    if not logger:
        raise Exception(
            "You're using wandb related callback, "
            "but WandbLogger was not found for some reason..."
        )

    return logger


class UploadCodeToWandbAsArtifact(Callback):
    """Upload all *.py files to wandb as an artifact at the beginning of the run."""

    def __init__(self, code_dir: str):
        self.code_dir = code_dir

    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        code = wandb.Artifact("project-source", type="code")
        for path in glob.glob(os.path.join(self.code_dir, "**/*.py"), recursive=True):
            code.add_file(path)

        experiment.use_artifact(code)


class UploadCheckpointsToWandbAsArtifact(Callback):
    """Upload experiment checkpoints to wandb as an artifact at the end of training."""

    def __init__(self, ckpt_dir: str = "checkpoints/", upload_best_only: bool = False):
        self.ckpt_dir = ckpt_dir
        self.upload_best_only = upload_best_only

    def on_train_end(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        ckpts = wandb.Artifact("experiment-ckpts", type="checkpoints")

        if self.upload_best_only:
            ckpts.add_file(trainer.checkpoint_callback.best_model_path)
        else:
            for path in glob.glob(
                os.path.join(self.ckpt_dir, "**/*.ckpt"), recursive=True
            ):
                ckpts.add_file(path)

        experiment.use_artifact(ckpts)


class WatchModelWithWandb(Callback):
    """Make WandbLogger watch model at the beginning of the run."""

    def __init__(self, log: str = "gradients", log_freq: int = 100):
        self.log = log
        self.log_freq = log_freq

    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        logger.watch(model=trainer.model, log=self.log, log_freq=self.log_freq)


class LogF1PrecisionRecallHeatmapToWandb(Callback):
    """
    Generate f1, precision and recall heatmap from validation step outputs.
    Expects validation step to return predictions and targets.
    Works only for single label classification!
    """

    def __init__(self, class_names: List[str] = None):
        self.class_names = class_names
        self.preds = []
        self.targets = []
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Gather data from single batch."""
        if self.ready:
            preds, targets = outputs["preds"], outputs["targets"]
            self.preds.append(preds)
            self.targets.append(targets)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate f1, precision and recall heatmap."""
        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment

            self.preds = torch.cat(self.preds).cpu()
            self.targets = torch.cat(self.targets).cpu()
            f1 = f1_score(self.preds, self.targets, average=None)
            r = recall_score(self.preds, self.targets, average=None)
            p = precision_score(self.preds, self.targets, average=None)

            experiment.log(
                {
                    f"f1_p_r_heatmap/{trainer.current_epoch}_{experiment.id}": wandb.plots.HeatMap(
                        x_labels=self.class_names,
                        y_labels=["f1", "precision", "recall"],
                        matrix_values=[f1, p, r],
                        show_text=True,
                    )
                },
                commit=False,
            )

            self.preds = []
            self.targets = []


class LogConfusionMatrixToWandb(Callback):
    """
    Generate Confusion Matrix.
    Expects validation step to return predictions and targets.
    Works only for single label classification!
    """

    def __init__(self, class_names: List[str] = None):
        self.class_names = class_names
        self.preds = []
        self.targets = []
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Gather data from single batch."""
        if self.ready:
            preds, targets = outputs["preds"], outputs["targets"]
            self.preds.append(preds)
            self.targets.append(targets)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate confusion matrix."""
        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment

            self.preds = torch.cat(self.preds).tolist()
            self.targets = torch.cat(self.targets).tolist()

            experiment.log(
                {
                    f"confusion_matrix/{trainer.current_epoch}_{experiment.id}": wandb.plot.confusion_matrix(
                        preds=self.preds,
                        y_true=self.targets,
                        class_names=self.class_names,
                    )
                },
                commit=False,
            )

            self.preds = []
            self.targets = []


''' BUGGED :(
class LogBestMetricScoresToWandb(Callback):
    """
    Store in wandb:
        - max train acc
        - min train loss
        - max val acc
        - min val loss
    Useful for comparing runs in table views, as wandb doesn't currently support column aggregation.
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
            experiment = logger.experiment

            metrics = trainer.callback_metrics

            if not self.train_loss_best or metrics["train/loss"] < self.train_loss_best:
                self.train_loss_best = metrics["train_loss"]

            if not self.train_acc_best or metrics["train/acc"] > self.train_acc_best:
                self.train_acc_best = metrics["train/acc"]

            if not self.val_loss_best or metrics["val/loss"] < self.val_loss_best:
                self.val_loss_best = metrics["val/loss"]

            if not self.val_acc_best or metrics["val/acc"] > self.val_acc_best:
                self.val_acc_best = metrics["val/acc"]

            experiment.log({"train/loss_best": self.train_loss_best}, commit=False)
            experiment.log({"train/acc_best": self.train_acc_best}, commit=False)
            experiment.log({"val/loss_best": self.val_loss_best}, commit=False)
            experiment.log({"val/acc_best": self.val_acc_best}, commit=False)
'''
