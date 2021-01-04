# pytorch lightning imports
import pytorch_lightning as pl

# normal imports
from typing import List, Tuple
import importlib.util
import os

# template utils imports
from pytorch_lightning import callbacks as lightning_callbacks
from pytorch_modules.lightning_callbacks import wandb_callbacks, custom_callbacks as custom_callbacks
from pytorch_lightning.loggers import CSVLogger


def format_path(path: str, base_dir: str):
    """Convert path to absolute relatively to base_dir and normalize it."""
    if not os.path.isabs(path):
        path = os.path.join(base_dir, path)
    return os.path.normpath(path)


def format_config_paths(config: dict, base_dir: str) -> dict:
    config["paths"]["data_dir"] = format_path(config["paths"]["data_dir"], base_dir)
    config["paths"]["logs_dir"] = format_path(config["paths"]["logs_dir"], base_dir)
    return config


def load_object(obj_path):
    obj_path, obj_name = obj_path.rsplit('.', 1)
    module = importlib.import_module(obj_path)

    assert hasattr(module, obj_name), \
        f'Object `{obj_name}` cannot be loaded from `{obj_path}`.'

    return getattr(module, obj_name)


def init_model(model_config: dict) -> pl.LightningModule:
    """
    Load LightningModule from path specified in run config.
    """
    model_class = model_config["class"]
    model_hparams = model_config["hparams"]

    LitModel = load_object(model_class)
    assert issubclass(LitModel, pl.LightningModule), \
        f"Specified model class `{model_class}` is not a subclass of `LightningModule`."

    return LitModel(hparams=model_hparams)


def init_datamodule(datamodule_config: dict, data_dir: str) -> pl.LightningDataModule:
    """
    Load LightningDataModule from path specified in run config.
    """
    datamodule_class = datamodule_config["class"]
    datamodule_hparams = datamodule_config["hparams"]

    DataModule = load_object(datamodule_class)

    assert issubclass(DataModule, pl.LightningDataModule), \
        f"Specified datamodule class `{datamodule_class}` is not a subclass of `LightningDataModule`"

    datamodule = DataModule(data_dir=data_dir, hparams=datamodule_hparams)
    datamodule.prepare_data()
    datamodule.setup()

    return datamodule


def init_trainer(project_config: dict,
                 run_config: dict,
                 callbacks: List[pl.Callback],
                 loggers: List[pl.loggers.LightningLoggerBase]) -> pl.Trainer:
    """
    Initialize PyTorch Lightning Trainer.
    """

    # Get path to checkpoint that you want to resume with if it was set in the run config
    resume_from_checkpoint = run_config.get("resume_training", {}).get("checkpoint_path", None)

    trainer = pl.Trainer(
        # whether to use gpu and how many
        gpus=project_config["num_of_gpus"],

        # experiment logging
        logger=loggers,

        # useful lightning_callbacks
        callbacks=callbacks,

        # resume training from checkpoint if it was set in the run config
        resume_from_checkpoint=resume_from_checkpoint
        if resume_from_checkpoint != "None"
        and resume_from_checkpoint != "False"
        and resume_from_checkpoint is not False
        else None,

        # print related
        progress_bar_refresh_rate=project_config["printing"]["progress_bar_refresh_rate"],
        profiler=SimpleProfiler() if project_config["printing"]["profiler"] else None,
        weights_summary=project_config["printing"]["weights_summary"],

        # number of validation sanity checks
        num_sanity_val_steps=3,

        # default log dir if no loggers is found
        default_root_dir=os.path.join(project_config["logs_dir"], "lightning_logs"),

        # insert all other trainer args specified in run config
        **run_config["trainer"]["args"]
    )

    return trainer


def init_callbacks(project_config: dict,
                   run_config: dict,
                   use_wandb: bool,
                   base_dir: str) -> List[pl.Callback]:
    """
    Initialize default lightning_callbacks and lightning_callbacks specified in run config.
    """

    default_callbacks = project_config.get("default_callbacks", {})
    run_callbacks = run_config.get("lightning_callbacks", {})

    # namespaces in which lightning_callbacks will be searched for
    namespaces = [
        lightning_callbacks,
        custom_callbacks,
        wandb_callbacks,
    ]

    callbacks = []

    all_callback_configs = {}
    for callback_config in default_callbacks:
        all_callback_configs[callback_config] = default_callbacks[callback_config]
    for callback_config in run_callbacks:
        all_callback_configs[callback_config] = run_callbacks[callback_config]

    for callback_config in all_callback_configs:
        found = False
        for namespace in namespaces:
            if hasattr(namespace, callback_config):
                callback_class = getattr(namespace, callback_config)
                callbacks.append(callback_class(**all_callback_configs[callback_config]))
                found = True
                break
        if not found:
            raise ModuleNotFoundError(f"Callback '{callback_config}' not found.")

    if use_wandb:
        callbacks.append(
            wandb_callbacks.SaveCodeToWandbCallback(
                base_dir=base_dir,
                wandb_save_dir=project_config["logs_dir"],
                run_config=run_config
            )
        )

    return callbacks


def init_loggers(project_config: dict,
                 run_config: dict,
                 model: pl.LightningModule,
                 datamodule: pl.LightningDataModule,
                 use_wandb: bool) -> List[pl.loggers.LightningLoggerBase]:
    """
    Initialize loggers.
    """
    loggers = []
    if use_wandb:
        wandb_logger = init_wandb_logger(
            project_config=project_config,
            run_config=run_config,
            model=model,
            datamodule=datamodule
        )
        if wandb_logger:
            loggers.append(wandb_logger)

    return loggers


def init_wandb_logger(project_config: dict,
                      run_config: dict,
                      model: pl.LightningModule,
                      datamodule: pl.LightningDataModule) -> pl.loggers.WandbLogger:
    """
    Initialize Weights&Biases loggers.
    """

    if "loggers" not in project_config or "wandb" not in project_config["loggers"]:
        return None

    # with this line wandb will throw an error if the run to be resumed does not exist yet
    # instead of auto-creating a new run
    os.environ["WANDB_RESUME"] = "must"

    resume_from_checkpoint = run_config.get("resume_training", {}).get("resume_from_checkpoint", None)
    wandb_run_id = run_config.get("resume_training", {}).get("wandb_run_id", None)

    wandb_logger = WandbLogger(
        project=project_config["loggers"]["wandb"]["project"],
        entity=project_config["loggers"]["wandb"]["entity"],
        log_model=project_config["loggers"]["wandb"]["log_model"],
        offline=project_config["loggers"]["wandb"]["offline"],

        group=run_config.get("wandb", {}).get("group", None),
        job_type=run_config.get("wandb", {}).get("job_type", "train"),
        tags=run_config.get("wandb", {}).get("tags", []),
        notes=run_config.get("wandb", {}).get("notes", ""),

        # resume run only if ckpt was set in the run config
        id=wandb_run_id
        if resume_from_checkpoint != "None" and wandb_run_id != "None" and resume_from_checkpoint is not None
        and resume_from_checkpoint is not False and wandb_run_id is not False
        else None,

        save_dir=project_config["logs_dir"],
        save_code=False
    )

    if not os.path.exists(project_config["logs_dir"]):
        os.makedirs(project_config["logs_dir"])

    if hasattr(model, 'model'):
        if project_config["loggers"]["wandb"]["log_gradients"]:
            wandb_logger.watch(model.model, log='gradients')
        else:
            wandb_logger.watch(model.model, log=None)
        wandb_logger.log_hyperparams({"architecture": model.model.__class__.__name__})
    else:
        if project_config["loggers"]["wandb"]["log_gradients"]:
            wandb_logger.watch(model, log='gradients')
        else:
            wandb_logger.watch(model, log=None)

    wandb_logger.log_hyperparams({
        "optimizer": model.configure_optimizers().__class__.__name__,
        "train_size": len(datamodule.data_train)
        if hasattr(datamodule, 'data_train') and datamodule.data_train is not None else 0,
        "val_size": len(datamodule.data_val)
        if hasattr(datamodule, 'data_val') and datamodule.data_val is not None else 0,
        "test_size": len(datamodule.data_test)
        if hasattr(datamodule, 'data_test') and datamodule.data_test is not None else 0,
    })
    wandb_logger.log_hyperparams({"num_of_gpus": project_config["num_of_gpus"]})
    wandb_logger.log_hyperparams(run_config["trainer"]["args"])
    wandb_logger.log_hyperparams(run_config["model"]["hparams"])
    wandb_logger.log_hyperparams(run_config["model"]["load_from"])
    wandb_logger.log_hyperparams(run_config["datamodule"]["hparams"])
    wandb_logger.log_hyperparams(run_config["datamodule"]["load_from"])

    return wandb_logger


def init_tensorboard_logger() -> pl.loggers.TensorBoardLogger:
    """Initialize tensorboard loggers"""
    # TODO
    return None


def init_comet_logger(project_config: dict,
                      run_config: dict,
                      model: pl.LightningModule,
                      datamodule: pl.LightningDataModule) -> pl.loggers.CometLogger:

    comet_logger = CometLogger()
