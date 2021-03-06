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


echo "TEST 1"
echo "fast_dev_run=true (run 1 train, val and test loop using 1 batch)"
python train.py trainer.fast_dev_run=True \
print_config=false

echo "TEST 2"
echo "Overfit to 10 bathes"
python train.py +trainer.overfit_batches=10 \
trainer.min_epochs=10 trainer.max_epochs=10 \
print_config=false

echo "TEST 3"
echo "Train 1 epoch on CPU"
python train.py trainer.gpus=0 trainer.max_epochs=1 \
print_config=false

echo "TEST 4"
echo "Train on 25% of data"
python train.py trainer.max_epochs=1 \
+trainer.limit_train_batches=0.25 +trainer.limit_val_batches=0.25 +trainer.limit_test_batches=0.25 \
print_config=false

echo "TEST 5"
echo "Train on 15 train batches, 10 val batches, 5 test batches"
python train.py trainer.max_epochs=1 \
+trainer.limit_train_batches=15 +trainer.limit_val_batches=10 +trainer.limit_test_batches=5 \
print_config=false

echo "TEST 6"
echo "Run all experiment configs for 2 epochs"
python train.py -m '+experiment=glob(*)' trainer.max_epochs=2 \
print_config=false

echo "TEST 7"
echo "Run default hydra sweep (executes 4 different combinations with fast_dev_run=True)"
python train.py -m datamodule.batch_size=32,64 model.lr=0.001,0.003 \
trainer.fast_dev_run=True \
print_config=false

echo "TEST 8"
echo "Run with 16 bit precision"
python train.py trainer.max_epochs=1 +trainer.precision=16 \
print_config=false

echo "TEST 9"
echo "Run with gradient accumulation"
python train.py trainer.max_epochs=1 +trainer.accumulate_grad_batches=10 \
print_config=false

echo "TEST 10"
echo "Run validation loop twice per epoch"
python train.py trainer.max_epochs=2 +trainer.val_check_interval=0.5 \
print_config=false

echo "TEST 11"
echo "Run with CSVLogger (2 epochs)"
python train.py logger=csv trainer.min_epochs=5 trainer.max_epochs=2 \
print_config=false

echo "TEST 12"
echo "Run with TensorBoardLogger (2 epochs)"
python train.py logger=tensorboard trainer.min_epochs=5 trainer.max_epochs=2 \
print_config=false
