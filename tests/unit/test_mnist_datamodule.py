import os
import torch
from src.datamodules.mnist_datamodule import MNISTDataModule


def test_mnist_datamodule():
    datamodule = MNISTDataModule()
    datamodule.prepare_data()

    assert not datamodule.data_train and not datamodule.data_val and not datamodule.data_test

    assert os.path.exists(os.path.join("data", "FashionMNIST"))
    assert os.path.exists(os.path.join("data", "FashionMNIST", "raw"))
    assert os.path.exists(os.path.join("data", "FashionMNIST", "processed"))

    datamodule.setup()

    assert datamodule.data_train and datamodule.data_val and datamodule.data_test
    assert len(datamodule.data_train) + len(datamodule.data_val) + len(datamodule.data_test) == 70_000

    assert datamodule.train_dataloader()
    assert datamodule.val_dataloader()
    assert datamodule.test_dataloader()

    batch = next(iter(datamodule.train_dataloader()))
    x, y = batch

    assert x.dtype == torch.float32
    assert y.dtype == torch.int64
