from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pytorch_lightning as pl
from shutil import copy
import inspect
import torch
import wandb
import os


class ExampleCallback(pl.Callback):
    def __init__(self):
        pass

    def on_init_start(self, trainer):
        print('Starting to initialize trainer!')

    def on_init_end(self, trainer):
        print('Trainer is initialized now.')

    def on_train_end(self, trainer, pl_module):
        print('Do something when training ends.')


class SaveOnnxModelToWandbCallback(pl.Callback):
    """
        Save model in .onnx format and upload to wandb.
        Might crash since not all models are compatible with onnx.
    """
    def __init__(self, datamodule, wandb_save_dir):
        first_batch = next(iter(datamodule.train_dataloader()))
        x, y = first_batch
        self.dummy_input = x
        self.wandb_save_dir = wandb_save_dir

    def on_sanity_check_end(self, trainer, pl_module):
        self.save_onnx_model(pl_module)

    def on_train_end(self, trainer, pl_module):
        self.save_onnx_model(pl_module)

    def save_onnx_model(self, pl_module):
        file_path = os.path.join(self.wandb_save_dir, "model.onnx")
        pl_module.to_onnx(file_path=file_path, input_sample=self.dummy_input.to(pl_module.device))
        wandb.save(file_path, base_path=self.wandb_save_dir)


class ImagePredictionLoggerCallback(pl.Callback):
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


class UnfreezeModelCallback(pl.Callback):
    """
        Unfreeze model after a few epochs.
    """
    def __init__(self, wait_epochs=5):
        self.wait_epochs = wait_epochs

    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch == self.wait_epochs:
            for param in pl_module.model.model.parameters():
                param.requires_grad = True


class MetricsHeatmapLoggerCallback(pl.Callback):
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
                    x_labels=self.class_names,
                    y_labels=["f1", "precision", "recall"],
                    matrix_values=[f1, p, r],
                    show_text=True,
                )}, commit=False)

            self.preds = []
            self.targets = []


class ConfusionMatrixLoggerCallback(pl.Callback):
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
            preds, targets = outputs
            self.preds.append(preds)
            self.targets.append(targets)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate f1, precision and recall heatmap."""
        if self.ready:
            self.preds = torch.cat(self.preds).tolist()
            self.targets = torch.cat(self.targets).tolist()

            trainer.logger.experiment.log({
                f"conf_mat{trainer.current_epoch}": wandb.plot.confusion_matrix(
                    self.preds,
                    self.targets,
                    class_names=self.class_names)
            }, commit=False)

            self.preds = []
            self.targets = []


class SaveCodeToWandbCallback(pl.Callback):
    """
        Upload specified code files to wandb at the beginning of the run.
    """
    def __init__(self, base_dir, wandb_save_dir, run_config):
        self.base_dir = base_dir
        self.wandb_save_dir = wandb_save_dir
        self.model_folder = run_config["model"]["model_folder"]
        self.datamodule_folder = run_config["dataset"]["datamodule_folder"]
        self.additional_files_to_be_saved = [  # paths should be relative to base_dir
            "utils/callbacks.py",
            "utils/init_utils.py",
            "train.py",
            "project_config.yaml",
            "run_configs.yaml",
        ]

    def on_sanity_check_end(self, trainer, pl_module):
        """Upload files when all validation sanity checks end."""
        # upload additional files
        for file in self.additional_files_to_be_saved:
            src_path = os.path.join(self.base_dir, file)
            dst_path = os.path.join(self.wandb_save_dir, "code", file)

            path, filename = os.path.split(dst_path)
            if not os.path.exists(path):
                os.makedirs(path)

            copy(src_path, dst_path)
            wandb.save(dst_path, base_path=self.wandb_save_dir)  # this line is only to make upload immediate

        # upload model files
        for filename in os.listdir(os.path.join(self.base_dir, "models", self.model_folder)):

            if filename.endswith('.py'):
                src_path = os.path.join(self.base_dir, "models", self.model_folder, filename)
                dst_path = os.path.join(self.wandb_save_dir, "code", "models", self.model_folder, filename)

                path, filename = os.path.split(dst_path)
                if not os.path.exists(path):
                    os.makedirs(path)

                copy(src_path, dst_path)
                wandb.save(dst_path, base_path=self.wandb_save_dir)  # this line is only to make upload immediate

        # upload datamodule files
        for filename in os.listdir(os.path.join(self.base_dir, "data_modules", self.datamodule_folder)):

            if filename.endswith('.py'):
                src_path = os.path.join(self.base_dir, "data_modules", self.datamodule_folder, filename)
                dst_path = os.path.join(self.wandb_save_dir, "code", "data_modules", self.datamodule_folder, filename)

                path, filename = os.path.split(dst_path)
                if not os.path.exists(path):
                    os.makedirs(path)

                copy(src_path, dst_path)
                wandb.save(dst_path, base_path=self.wandb_save_dir)  # this line is only to make upload immediate
