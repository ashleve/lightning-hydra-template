import glob
import os
from typing import List

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning import Callback
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import f1_score, precision_score, recall_score


def get_wandb_logger(trainer: pl.Trainer) -> WandbLogger:
    logger = None
    for lg in trainer.logger:
        if isinstance(lg, WandbLogger):
            logger = lg

    if not logger:
        raise Exception(
            "You are using wandb related callback,"
            "but WandbLogger was not found for some reason..."
        )

    return logger


class UploadCodeToWandbAsArtifact(Callback):
    """Upload all *.py files to wandb as an artifact, at the beginning of the run."""

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
    """Upload checkpoints to wandb as an artifact, at the end of training."""

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
