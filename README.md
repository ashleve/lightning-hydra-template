# Hackathon template
A convenient starting template for any deep learning hackathon. Built with PyTorch Lightning and Weights&Biases(wandb).


### Create conda env
```
    conda create --name hack_env
    conda activate hack_env
```
### Install pytorch with conda
https://pytorch.org/get-started/locally/
```
    conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
```
### Clone repo
```
    git clone https://github.com/kinoai/hackathon-template
    cd hackathon-template
    pip install -r requirements.txt
```

### Useful tips
- PyTorch Lightning Bolts is official collection of prebuilt models across many research domains:
https://pytorch-lightning.readthedocs.io/en/latest/bolts.html

- Useful pl.Trainer() parameters:
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
    - <b>gradient_clip_val=0.5</b> - gradient clipping value (0 means don’t clip), helps with exploding gradient issues
    - <b>profiler=SimpleProfiler()</b> - print execution time info for each method used
    - <b>weights_summary='full'</b> - print model info
    - <b>amp_backend='apex'</b> - apex backend for mixed precision training https://github.com/NVIDIA/apex

