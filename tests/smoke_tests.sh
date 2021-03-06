# !/bin/bash
# These are just a couple of quick experiments to test if your model doesn't crash under different conditions

# To execute:
# bash tests/quick_tests.sh

# Method for printing test name
echo() {
  termwidth="$(tput cols)"
  padding="$(printf '%0.1s' ={1..500})"
  printf '\e[33m%*.*s %s %*.*s\n\e[0m' 0 "$(((termwidth-2-${#1})/2))" "$padding" "$1" 0 "$(((termwidth-1-${#1})/2))" "$padding"
}

# Make python hide warnings
export PYTHONWARNINGS="ignore"


# Test fast_dev_run (runs for 1 train, 1 val and 1 test batch)
echo "TEST 1"
python train.py +trainer.fast_dev_run=True \
print_config=false

# Overfit to 10 bathes
echo "TEST 2"
python train.py +trainer.overfit_batches=10 \
trainer.min_epochs=20 trainer.max_epochs=20 \
print_config=false

# Test 1 epoch on CPU
echo "TEST 3"
python train.py trainer.gpus=0 trainer.max_epochs=1 \
print_config=false

# Test 1 epoch on GPU
echo "TEST 4"
python train.py trainer.gpus=1 trainer.max_epochs=1 \
print_config=false

# Test on 25% of data
echo "TEST 5"
python train.py trainer.max_epochs=1 \
+trainer.limit_train_batches=0.25 +trainer.limit_val_batches=0.25 +trainer.limit_test_batches=0.25 \
print_config=false

# Test on 15 train batches, 10 val batches, 5 test batches
echo "TEST 6"
python train.py trainer.max_epochs=1 \
+trainer.limit_train_batches=15 +trainer.limit_val_batches=10 +trainer.limit_test_batches=5 \
print_config=false

# Test all experiment configs
echo "TEST 7"
python train.py -m '+experiment=glob(*)' trainer.gpus=1 trainer.max_epochs=2 \
print_config=false

# Test default hydra sweep over hyperparameters (runs 4 different combinations with fast_dev_run)
echo "TEST 8"
python train.py -m datamodule.batch_size=32,64 model.lr=0.001,0.003 \
+trainer.fast_dev_run=True \
print_config=false

# Test multiple workers and cuda pinned memory
echo "TEST 9"
python train.py trainer.gpus=1 trainer.max_epochs=2 \
datamodule.num_workers=4 datamodule.pin_memory=True \
print_config=false 

# Test 16 bit precision
echo "TEST 10"
python train.py trainer.gpus=1 trainer.max_epochs=1 precision=16 \
print_config=false

# Test gradient accumulation
echo "TEST 11"
python train.py trainer.gpus=1 trainer.max_epochs=1 accumulate_grad_batches=10 \
print_config=false

# Test running validation loop twice per epoch
echo "TEST 12"
python train.py trainer.gpus=1 trainer.max_epochs=2 val_check_interval=0.5 \
print_config=false

# Test CSV logger (5 epochs)
echo "TEST 13"
python train.py logger=csv_logger trainer.min_epochs=5 trainer.max_epochs=5 trainer.gpus=1 \
print_config=false

# Test TensorBoard logger (5 epochs)
echo "TEST 14"
python train.py logger=tensorboard trainer.min_epochs=5 trainer.max_epochs=5 trainer.gpus=1 \
print_config=false

# Test mixed-precision training
echo "TEST 15"
python train.py trainer.gpus=1 trainer.max_epochs=3 \
+amp_backend='apex' amp_level='O2' \
print_config=false
