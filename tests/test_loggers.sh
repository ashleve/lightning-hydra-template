# Test CSV logger
python train.py logger=csv_logger trainer.min_epochs=3 trainer.max_epochs=3

# Test Weights&Biases logger
python train.py logger=wandb logger.wandb.project_name="template_tests" trainer.min_epochs=10 trainer.max_epochs=10

# Test TensorBoard logger
python train.py logger=tensorboard trainer.min_epochs=10 trainer.max_epochs=10
