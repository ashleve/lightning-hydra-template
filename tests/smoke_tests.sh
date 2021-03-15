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
echo "Debug mode (run 1 train, val and test loop using 1 batch)"
python run.py trainer=debug_trainer debug=True \
print_config=false

echo "TEST 2"
echo "Overfit to 10 batches (10 epochs)"
python run.py trainer=debug_trainer trainer.overfit_batches=10 \
trainer.min_epochs=10 trainer.max_epochs=10 \
print_config=false

echo "TEST 3"
echo "Train on CPU (1 epoch)"
python run.py trainer=debug_trainer trainer.gpus=0 trainer.max_epochs=1 \
print_config=false

echo "TEST 4"
echo "Train on 25% of data (1 epoch)"
python run.py trainer=debug_trainer trainer.max_epochs=1 \
trainer.limit_train_batches=0.25 trainer.limit_val_batches=0.25 trainer.limit_test_batches=0.25 \
print_config=false

echo "TEST 5"
echo "Train on 15 train batches, 10 val batches, 5 test batches (1 epoch)"
python run.py trainer=debug_trainer trainer.max_epochs=1 \
trainer.limit_train_batches=15 trainer.limit_val_batches=10 trainer.limit_test_batches=5 \
print_config=false

echo "TEST 6"
echo "Run all experiment configs (2 epochs)"
python run.py -m trainer=debug_trainer '+experiment=glob(*)' trainer.max_epochs=2 \
print_config=false

echo "TEST 7"
echo "Run default hydra sweep (executes 4 different combinations in debug mode)"
python run.py -m trainer=debug_trainer datamodule.batch_size=32,64 optimizer.lr=0.001,0.003 \
debug=True \
print_config=false

echo "TEST 8"
echo "Run with gradient accumulation (1 epoch)"
python run.py trainer=debug_trainer trainer.max_epochs=1 trainer.accumulate_grad_batches=10 \
print_config=false

echo "TEST 9"
echo "Run validation loop twice per epoch (1 epoch)"
python run.py trainer=debug_trainer trainer.max_epochs=1 trainer.val_check_interval=0.5 \
print_config=false

echo "TEST 10"
echo "Run with CSVLogger (3 epochs)"
python run.py trainer=debug_trainer logger=csv trainer.max_epochs=2 trainer.limit_train_batches=10 \
print_config=false

echo "TEST 11"
echo "Run with TensorBoardLogger (3 epochs)"
python run.py trainer=debug_trainer logger=tensorboard trainer.max_epochs=2 trainer.limit_train_batches=10 \
print_config=false
