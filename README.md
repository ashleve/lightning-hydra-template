# Hackathon template
A convenient starting template for most deep learning projects. Built with PyTorch Lightning and Weights&Biases (wandb).


## Setup
Read [SETUP.md](SETUP.md)


## Project structure
The directory structure of new project looks like this: 
```
├── project
│   ├── data                <- Data from third party sources
│   │
│   ├── docs                <- Useful pdf files
│   │
│   ├── hack_utils          <- Any extra scripts not belonging to training pipeline
│   │
│   ├── notebooks           <- Jupyter notebooks
│   │
│   ├── training_modules    <- All modules useful for training deep learning models
│   │   ├── callbacks.py            <- Useful training callbacks
│   │   ├── datasets.py             <- PyTorch "Dataset" modules
│   │   ├── datamodules.py          <- "LightningDataModule" modules (wrappers for PyTorch "Dataset")
│   │   ├── lightning_wrapper.py    <- Contains train/val/test step methods executed during training
│   │   ├── loggers.py              <- Initializers for different loggers (wandb, tensorboard, etc.)
│   │   ├── models.py               <- Neural networks declarations
│   │   └── transforms.py           <- Data transformations (data preprocessing)
│   │
│   ├── config.yaml         <- Training configuration
│   ├── execute_sweep.py    <- Special file for executing wandb sweeps (hyperparameter search)
│   ├── predict.py          <- Make predictions with trained model
│   └── train.py            <- Train model
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```


## Config parameters explanation 
#### [config.yaml](project/config.yaml):
```yaml
num_of_gpus: -1                     <- -1 means use all gpus available on your machine, 0 means train on cpu

hparams:                            <- you can add any parameters here and then acces them in your network model or datamodule, those parameters are always saved to wandb config
    max_epochs: 3
    batch_size: 64
    lr: 0.001
    weight_decay: 0.000001                  <- L2 normalization set in optimizer
    gradient_clip_val: 0.5                  <- gradient clipping value (0 means don’t clip), helps with exploding gradient issues
    accumulate_grad_batches: 1              <- perform optimisation after accumulating gradient from n batches

resume:
    resume_from_ckpt: False         <- set to True if you want to resume
    wandb_run_id: "8uuomodb"        <- id of wandb run you want to resume
    ckpt_path: "epoch=2.ckpt"       <- lightning checkpoint path

loggers:
    wandb:
        project: "hackathon_template_test"  <- wandb project name
        team: "kino"                        <- entity name (your username or team name you belong to)
        group: None
        job_type: "train"
        tags: []
        log_model: True                     <- True if you want to automatically upload your model to wandb at the end of training
        offline: False                      <- True if you don't want to send any data to wandb server

callbacks:
    checkpoint:
        monitor: "val_acc"                  <- name of the logged metric that determines when ckpt is saved
        save_top_k: 1                       <- save k best models (determined by above metric)
        save_last: True                     <- additionaly always save model from last epoch
        mode: "max"                         <- determine whether improving means minimizing or maximizing metrics score (alternatively "min")
    early_stop:
        monitor: "val_acc"                  <- name of the logged metric that determines when training is stopped
        patience: 100                       <- for how long metric needs to not improve in order to stop training 
        mode: "max"                         <- determine whether improving means minimizing or maximizing metrics score (alternatively "min")

printing:
    progress_bar_refresh_rate: 5            <- refresh rate of training bar in terminal
    weights_summary: "top"                  <- print summary of model an the beginning of the run (alternatively "full")
    profiler: False                         <- True will print mean execution time of all methods at the end of the training

```


## Tips
See [USEFUL_TIPS.md](USEFUL_TIPS.md)
