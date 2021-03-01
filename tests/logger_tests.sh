# TESTS FOR DIFFERENT LOGGERS
# TO EXECUTE:
# bash tests/logger_tests.sh

# conda activate testenv

# Test CSV logger
echo TEST 1
python train.py logger=csv_logger trainer.min_epochs=3 trainer.max_epochs=3 trainer.gpus=1

# # Test Weights&Biases logger
echo TEST 2
python train.py logger=wandb logger.wandb.project="env_tests" trainer.min_epochs=10 trainer.max_epochs=10 trainer.gpus=1

# Test TensorBoard logger
echo TEST 3
python train.py logger=tensorboard trainer.min_epochs=10 trainer.max_epochs=10 trainer.gpus=1

# Test many loggers at once
echo TEST 4
python train.py logger=many_loggers trainer.min_epochs=10 trainer.max_epochs=10 trainer.gpus=1
