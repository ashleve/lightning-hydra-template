from pytorch_lightning.callbacks import Callback
import torch
import wandb
import os


class ExampleCallback(Callback):

    def on_init_start(self, trainer):
        print('Starting to initialize trainer!')

    def on_init_end(self, trainer):
        print('Trainer is initialized now.')

    def on_train_end(self, trainer, pl_module):
        print('Do something when training ends.')


class SaveOnnxToWandbCallback(Callback):

    def __init__(self, dataloader, wandb_save_dir):
        first_batch = next(iter(dataloader))
        x, y = first_batch
        self.dummy_input = x
        self.save_dir = wandb_save_dir

    def on_sanity_check_end(self, trainer, pl_module):
        self.save_onnx_model(pl_module)

    def on_train_end(self, trainer, pl_module):
        self.save_onnx_model(pl_module)

    def save_onnx_model(self, pl_module):
        file_path = os.path.join(self.save_dir, "model.onnx")
        pl_module.to_onnx(file_path=file_path, input_sample=self.dummy_input.to(pl_module.device))
        wandb.save(file_path)


class ImagePredictionLoggerCallback(Callback):
    def __init__(self, val_samples, num_samples=32):
        super().__init__()
        self.val_imgs, self.val_labels = val_samples[:num_samples]

    def on_validation_epoch_end(self, trainer, pl_module):
        val_imgs = self.val_imgs.to(device=pl_module.device)

        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, -1)

        trainer.logger.experiment.log({
            "examples": [wandb.Image(x, caption=f"Pred:{pred}, Label:{y}")
                         for x, pred, y in zip(val_imgs, preds, self.val_labels)],
            "global_step": trainer.global_step
        })