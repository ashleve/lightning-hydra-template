from pytorch_lightning.callbacks import Callback
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


class SaveOnnxToWandbCallback(Callback):
    """Save model in .onnx format."""
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
    """Each epoch upload to wandb a couple of the same images with predicted labels."""
    def __init__(self, val_samples, num_samples=32):
        super().__init__()
        self.val_imgs, self.val_labels = val_samples[:num_samples]

    def on_validation_epoch_end(self, trainer, pl_module):
        val_imgs = self.val_imgs.to(device=pl_module.device)

        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, -1)

        trainer.logger.experiment.log({"examples": [
            wandb.Image(x, caption=f"Pred:{pred}, Label:{y}") for x, pred, y in zip(val_imgs, preds, self.val_labels)
        ]})


class UnFreezeModelCallback(Callback):
    """Unfreeze model after a few epochs."""
    def __init__(self, wait_epochs=5):
        self.wait_epochs = wait_epochs

    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch == self.wait_epochs:
            for param in pl_module.model.model.parameters():
                param.requires_grad = True


class ConfusionMatrixLoggerCallback(Callback):
    """Log confusion matrix calculated from each validation epoch."""
    def __init__(self):
        self.val_loop_outputs = []
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Gather data from single batch."""
        if self.ready:
            self.val_loop_outputs.append(outputs)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.ready:
            for outputs in self.val_loop_outputs:
                preds, targets = outputs
            self.val_loop_outputs = []
