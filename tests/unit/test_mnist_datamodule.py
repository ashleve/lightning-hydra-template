import os

import pytest
import torch

from src.datamodules.mnist_datamodule import MNISTDataModule


@pytest.mark.parametrize("batch_size", [32, 128])
def test_mnist_datamodule(data_dir="data/", batch_size=32):
    dm = MNISTDataModule(data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert os.path.exists(os.path.join(data_dir, "MNIST"))
    assert os.path.exists(os.path.join(data_dir, "MNIST", "raw"))

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 70_000

    assert dm.train_dataloader()
    assert dm.val_dataloader()
    assert dm.test_dataloader()

    batch = next(iter(dm.train_dataloader()))
    x, y = batch

    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64
