# Hackathon template
A convenient starting template for most deep learning projects. Built with PyTorch Lightning and Weights&Biases (wandb).


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

## Setup

#### 1. Install anaconda
https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html

#### 2. Create anaconda env
```
    conda create --name hack_env
    conda activate hack_env
```

#### 3. Make sure proper python PATH is loaded
Unix
```
    which python
```
Windows
```
    for %i in (python.exe) do @echo. %~$PATH:i
```
Expected result: `PATH_TO_CONDA/envs/ENV_NAME/bin/python`

#### 4. Install pytorch with conda
Installation command generator: https://pytorch.org/get-started/locally/
```
    conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
```

#### 5. Clone repo
```
    git clone https://github.com/kinoai/hackathon-template
```

#### 6. Install requirements with pip
```
    cd hackathon-template
    pip install -r requirements.txt
```

#### 7. Log into your wandb account
```
    wandb login
```

#### 8. PyCharm setup
- open this repository as PyCharm project
- set "hack_env" as project interpreter:<br> 
`Ctrl + Shift + A -> type "Project Interpreter"`
- mark folder "project" as sources root:<br>
`right click on directory -> "Mark Directory as" -> "Sources Root"`
- set terminal emulation:<br> 
`Ctrl + Shift + A -> type "Edit Configurations..." -> select "Emulate terminal in output console"`
- run training:<br>
`right click on train.py file -> "Run 'train'"`



#### Important notes!
- If you are not using GPU (CUDA incompatible GPU) you may need to specify the number of GPUs manually instead of leaving the default `-1` in `config.yaml`:
```
    num_of_gpus: 0
```
<br>


## [config.yaml](project/train_config.yaml) parameters explanation:
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
<br>


## Useful tips
- PyTorch Lightning Bolts is official collection of prebuilt models across many research domains:
    - https://pytorch-lightning.readthedocs.io/en/latest/bolts.html
    - https://github.com/PyTorchLightning/pytorch-lightning-bolts
    
- Pre-trained pytorch model repository designed for research exploration:
    - https://pytorch.org/hub/
    
- List of all tools in PyTorch ecosystem:
    - https://pytorch.org/ecosystem/

- Additional pl.Trainer() parameters which can be useful:
    - <b>accumulate_grad_batches=5</b> - perform optimisation after accumulating gradient from 5 batches
    - <b>accumulate_grad_batches={5: 3, 10: 20}</b> - no accumulation for epochs 1-4. accumulate 3 for epochs 5-10. accumulate 20 after that
    - <b>auto_scale_batch_size='power'</b> - automatically find the largest batch size that fits into memory and is power of 2 (requires calling trainer.tune(model, datamodule))
    - <b>check_val_every_n_epoch=10</b> - run validation loop every 10 training epochs
    - <b>val_check_interval=0.25</b> - check validation set 4 times during a training epoch
    - <b>fast_dev_run=True</b> - runs 1 train, val, test batch and program ends (great for debugging)
    - <b>min_epochs=1</b> - force training for at least these many epochs
    - <b>overfit_batches=0.01</b> - use only 1% of the train set (and use the train set for val and test)
    - <b>overfit_batches=10</b> - use only 10 batches of the train set (and use the train set for val and test)
    - <b>limit_train_batches=0.25</b> - run through only 25% of the training set each epoch
    - <b>limit_val_batches=0.25</b>
    - <b>limit_test_batches=0.25</b>
    - <b>precision=16</b> - set tensor precision (default is 32 bits)
    - <b>amp_backend='apex'</b> - apex backend for mixed precision training https://github.com/NVIDIA/apex
