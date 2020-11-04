# regular imports
import yaml

# pytorch lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.profiler import SimpleProfiler, AdvancedProfiler

# wandb
import wandb
from pytorch_lightning.loggers import WandbLogger

# custom utils
from utils.data_modules import MNISTDataModule
from utils.lightning_wrapper import LitModel
from utils.callbacks import ExampleCallback


# Load config
with open("config.yaml", "r") as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.FullLoader)


# Init our model
model = LitModel(config=config)


# Init data loader
dataloader = MNISTDataModule(batch_size=128)
dataloader.prepare_data()
dataloader.setup()


# Init wandb logger
wandb_logger = WandbLogger(
    project=config["loggers"]["wandb"]["project"],
    job_type=config["loggers"]["wandb"]["job_type"],
    tags=config["loggers"]["wandb"]["tags"],
    entity=config["loggers"]["wandb"]["team"],
    log_model=True,
    offline=False
)
wandb_logger.watch(model.model, log='all')


# Init callbacks
callbacks = [
    ExampleCallback(),
    EarlyStopping(
        monitor=config["callbacks"]["early_stop"]["monitor"],
        patience=config["callbacks"]["early_stop"]["patience"],
        mode=config["callbacks"]["early_stop"]["mode"],
        verbose=False
    ),
]

checkpoint_callback = ModelCheckpoint(
    monitor=config["callbacks"]["checkpoint"]["monitor"],
    save_top_k=config["callbacks"]["checkpoint"]["save_top_k"],
    mode=config["callbacks"]["checkpoint"]["mode"]
)


# Init trainer
trainer = pl.Trainer(
    gpus=config["num_of_gpus"],
    max_epochs=config["max_epochs"],
    logger=wandb_logger,
    callbacks=callbacks,
    checkpoint_callback=checkpoint_callback,
    resume_from_checkpoint=config["resume"]["ckpt_path"] if config["resume"]["resume_from_ckpt"] else None,
    auto_scale_batch_size='power' if config["auto_scale_batch_size"] else False,
    accumulate_grad_batches=config["accumulate_grad_batches"],
    gradient_clip_val=config["gradient_clip_val"],
    progress_bar_refresh_rate=1,
    profiler=SimpleProfiler(),
    weights_summary='full',
    # fast_dev_run=True,
    # limit_train_batches=0.01
    # limit_val_batches=0.01
    # limit_test_batches=0.01
    # amp_backend='apex',
    # precision=16,
)

# Tune trainer (finds biggest possible batch size if enabled)
if config["auto_scale_batch_size"]:
    trainer.tune(model=model, datamodule=dataloader)

# Train the model ⚡
trainer.fit(model=model, datamodule=dataloader)

# Evaluate model on test set
trainer.test()

# Close wandb run
wandb.finish()
