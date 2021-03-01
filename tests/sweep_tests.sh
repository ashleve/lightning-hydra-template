# TESTS FOR HYPERPARAMETER SWEEPS
# TO EXECUTE:
# bash tests/sweep_tests.sh

# conda activate testenv


# currently there are some issues with running sweeps alongside wandb 
# https://github.com/wandb/client/issues/1314
# this env variable fixes that
export WANDB_START_METHOD=thread


# Test default hydra sweep with wandb logging
echo TEST 1
python train.py -m datamodule.batch_size=64,128 model.lr=0.001,0.003 \
+experiment=exp_example_simple \
trainer.gpus=1 trainer.max_epochs=2 seed=12345 \
datamodule.num_workers=12 datamodule.pin_memory=True \
logger=wandb logger.wandb.project="env_tests" logger.wandb.group="DefaultSweep_MNIST_SimpleDenseNet"

# Test optuna sweep with wandb logging
echo TEST 2
python train.py -m --config-name config_optuna.yaml \
+experiment=exp_example_simple \
trainer.gpus=1 trainer.max_epochs=5 seed=12345 \
datamodule.num_workers=12 datamodule.pin_memory=True \
logger=wandb logger.wandb.project="env_tests" logger.wandb.group="Optuna_MNIST_SimpleDenseNet"
