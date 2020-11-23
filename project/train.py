from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.profiler import SimpleProfiler
import importlib
import yaml

# training modules
import training_modules.datamodules
from training_modules.callbacks import *
from training_modules.loggers import *


def train():
    # Load train config
    train_config = load_config(path="train_config.yaml")

    # Load model config
    model_config_path = os.path.join("models", train_config["model"], "config.yaml")
    model_config = load_config(path=model_config_path)

    # Init lightning model
    module_path = "models." + train_config["model"] + ".lightning_module"
    lit_model = importlib.import_module(module_path).LitModel(hparams=model_config["hparams"])

    # Init data module
    datamodule_name = train_config["data"]
    args = train_config["dataset_params"][datamodule_name]
    args["batch_size"] = model_config["hparams"]["batch_size"]
    datamodule = getattr(training_modules.datamodules, train_config["data"])(**args)
    datamodule.prepare_data()
    datamodule.setup()

    # Init experiment logger
    logger = get_wandb_logger(train_config, model_config, lit_model, datamodule)

    # Init callbacks
    callbacks = [
        EarlyStopping(
            monitor=train_config["callbacks"]["early_stop"]["monitor"],
            patience=train_config["callbacks"]["early_stop"]["patience"],
            mode=train_config["callbacks"]["early_stop"]["mode"],
        ),
        ModelCheckpoint(
            monitor=train_config["callbacks"]["checkpoint"]["monitor"],
            save_top_k=train_config["callbacks"]["checkpoint"]["save_top_k"],
            mode=train_config["callbacks"]["checkpoint"]["mode"],
            save_last=train_config["callbacks"]["checkpoint"]["save_last"],
        ),
        # MetricsHeatmapLoggerCallback(),
        # UnfreezeModelCallback(wait_epochs=5),
        # ImagePredictionLoggerCallback(datamodule=datamodule),
        # SaveCodeToWandbCallback(wandb_save_dir=logger.save_dir),
        # SaveOnnxModelToWandbCallback(datamodule=datamodule, save_dir=logger.save_dir)
    ]

    # Init trainer
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        gpus=train_config["num_of_gpus"],
        max_epochs=model_config["hparams"]["max_epochs"],
        resume_from_checkpoint=model_config["resume_training"]["lightning_ckpt"]["ckpt_path"]
        if model_config["resume_training"]["lightning_ckpt"]["resume_from_ckpt"] else None,
        accumulate_grad_batches=model_config["hparams"]["accumulate_grad_batches"],
        gradient_clip_val=model_config["hparams"]["gradient_clip_val"],
        progress_bar_refresh_rate=train_config["printing"]["progress_bar_refresh_rate"],
        profiler=SimpleProfiler() if train_config["printing"]["profiler"] else None,
        weights_summary=train_config["printing"]["weights_summary"],
        num_sanity_val_steps=3,
        default_root_dir="logs/lightning_logs"
        # fast_dev_run=True,
        # min_epochs=10,
        # limit_train_batches=0.1
        # limit_val_batches=0.01
        # limit_test_batches=0.01
        # auto_scale_batch_size="power",
        # amp_backend='apex',
        # precision=16,
    )

    # Test before training
    # trainer.test(model=model, datamodule=datamodule)

    # Save randomly initialized model
    # trainer.save_checkpoint("random.ckpt")

    # Train the model ⚡
    trainer.fit(model=lit_model, datamodule=datamodule)

    # Evaluate model on test set
    trainer.test()


def load_config(path):
    with open(path, "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return config


if __name__ == "__main__":
    train()
