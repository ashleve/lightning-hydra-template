from sklearn.metrics import precision_score, recall_score, f1_score
from pytorch_lightning.loggers import WandbLogger
from wandb.sdk.wandb_run import Run as wandb_run
from pytorch_lightning import Callback
import random
import torch
import wandb
import glob
import os


class SaveCodeToWandb(Callback):
    """
    Upload all *.py files to wandb at the beginning of the run.
    """
    def __init__(self, code_dir: str):
        self.code_dir = code_dir

    def on_sanity_check_end(self, trainer, pl_module):
        """Upload files when all validation sanity checks end."""
        code = wandb.Artifact('project-source', type='code')
        for path in glob.glob(os.path.join(self.code_dir, '**/*.py'), recursive=True):
            code.add_file(path)
        wandb.run.use_artifact(code)


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


class SaveMetricsHeatmapToWandb(Callback):
    """
    Generate f1, precision and recall heatmap from validation epoch outputs.
    Expects validation step to return predictions and targets.
    Works only for single label classification!
    """
    def __init__(self, class_names):
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
            preds, targets = outputs["preds"], outputs["y"]
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

            logger = None
            for some_logger in trainer.logger.experiment:
                if isinstance(some_logger, wandb_run):
                    logger = some_logger

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
    def __init__(self, class_names):
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
            preds, targets = outputs["preds"], outputs["y"]
            self.preds.append(preds)
            self.targets.append(targets)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate f1, precision and recall heatmap."""
        if self.ready:
            self.preds = torch.cat(self.preds).tolist()
            self.targets = torch.cat(self.targets).tolist()

            logger = None
            for some_logger in trainer.logger.experiment:
                if isinstance(some_logger, wandb_run):
                    logger = some_logger

            logger.log({
                f"conf_mat_{trainer.current_epoch}_{logger.id}": wandb.plot.confusion_matrix(
                    self.preds,
                    self.targets,
                    class_names=self.class_names)
            }, commit=False)

            self.preds = []
            self.targets = []


class SaveBestMetricScoresToWandb(Callback):
    def __init__(self):
        self.train_loss_list = []
        self.train_acc_list = []
        self.train_loss_best = None
        self.train_acc_best = None

        self.val_loss_list = []
        self.val_acc_list = []
        self.val_loss_best = None
        self.val_acc_best = None

        self.ready = False

    def clear_lists(self):
        self.train_loss_list.clear()
        self.train_acc_list.clear()
        self.val_loss_list.clear()
        self.val_acc_list.clear()

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Gather data from single batch."""
        # This is bugged :/ - lets wait for lightning 1.2 update
        # if self.ready:
        #     self.train_loss_list.append(outputs["loss"])
        #     self.train_acc_list.append(outputs["acc"])
        pass

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Gather data from single batch."""
        if self.ready:
            self.val_loss_list.append(outputs["loss"])
            self.val_acc_list.append(outputs["acc"])

    def on_epoch_end(self, trainer, pl_module):
        if self.ready:
            for logger in trainer.logger.experiment:
                if isinstance(logger, wandb_run):
                    # loss = sum(self.train_loss_list) / len(self.train_loss_list)
                    # acc = sum(self.train_acc_list) / len(self.train_acc_list)
                    # self.train_loss_best = loss if self.train_loss_best is None or loss < self.train_loss_best else self.train_loss_best
                    # self.train_acc_best = acc if self.train_acc_best is None or acc > self.train_acc_best else self.train_acc_best
                    # logger.log({"train_loss_best": self.train_loss_best}, commit=False)
                    # logger.log({"train_acc_best": self.train_acc_best}, commit=False)

                    loss = sum(self.val_loss_list) / len(self.val_loss_list)
                    acc = sum(self.val_acc_list) / len(self.val_acc_list)
                    self.val_loss_best = loss if self.val_loss_best is None or loss < self.val_loss_best else self.val_loss_best
                    self.val_acc_best = acc if self.val_acc_best is None or acc > self.val_acc_best else self.val_acc_best
                    logger.log({"val_loss_best": self.val_loss_best}, commit=False)
                    logger.log({"val_acc_best": self.val_acc_best}, commit=False)

        self.clear_lists()
