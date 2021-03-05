# !/bin/bash
# Test hyperparameter sweeps

# To execute:
# bash tests/sweep_tests.sh

echo() {
  termwidth="$(tput cols)"
  padding="$(printf '%0.1s' ={1..500})"
  printf '\e[33m%*.*s %s %*.*s\n\e[0m' 0 "$(((termwidth-2-${#1})/2))" "$padding" "$1" 0 "$(((termwidth-1-${#1})/2))" "$padding"
}

# Make python hide warnings
export PYTHONWARNINGS="ignore"

# Test default hydra sweep with wandb logging
echo "TEST 1"
python train.py -m datamodule.batch_size=64,128 model.lr=0.001,0.003 \
+experiment=exp_example_simple \
trainer.gpus=1 trainer.max_epochs=2 seed=12345 \
datamodule.num_workers=12 datamodule.pin_memory=True \
logger=wandb logger.wandb.project="env_tests" logger.wandb.group="DefaultSweep_MNIST_SimpleDenseNet"

# Test optuna sweep with wandb logging
echo "TEST 2"
python train.py -m --config-name config_optuna.yaml \
+experiment=exp_example_simple \
trainer.gpus=1 trainer.max_epochs=5 seed=12345 \
datamodule.num_workers=12 datamodule.pin_memory=True \
logger=wandb logger.wandb.project="env_tests" logger.wandb.group="Optuna_MNIST_SimpleDenseNet"
