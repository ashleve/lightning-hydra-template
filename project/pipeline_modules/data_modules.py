from pl_bolts.models.self_supervised.cpc.transforms import CPCTrainTransformsImageNet128
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST, CIFAR10
import pytorch_lightning as pl
import numpy as np

# custom
from pipeline_modules.transforms import *
from pipeline_modules.datasets import *


class ExampleDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="data/example_data", batch_size=64, split_ratio=0.90):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.split_ratio = split_ratio

        self.data_train = None
        self.data_val = None
        self.data_test = None

        # optional variables
        self.input_dims = None
        self.input_size = None

    def prepare_data(self):
        """Download data if needed."""
        pass

    def setup(self, stage=None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        pass

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size)


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='data/mnist', batch_size=64, split_ratio=0.90):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.split_ratio = split_ratio

        self.data_train = None
        self.data_val = None
        self.data_test = None

        self.input_dims = None
        self.input_size = None

    def prepare_data(self):
        """Download data if needed."""
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        trainset = MNIST(self.data_dir, train=True, transform=transforms.ToTensor())

        train_dataset_length = int(len(trainset) * self.split_ratio)
        train_val_split = [train_dataset_length, len(trainset) - train_dataset_length]

        self.data_train, self.data_val = random_split(trainset, train_val_split)
        self.data_test = MNIST(self.data_dir, train=False, transform=transforms.ToTensor())

        self.input_dims = self.data_train[0][0].shape
        self.input_size = np.prod(self.input_dims)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size)


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir="data/cifar10", batch_size=64, split_ratio=0.90):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.split_ratio = split_ratio

        self.data_train = None
        self.data_val = None
        self.data_test = None

        self.input_dims = None
        self.input_size = None

    def prepare_data(self):
        """Download data if needed."""
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        trainset = CIFAR10(self.data_dir, train=True, transform=transforms.ToTensor())

        train_dataset_length = int(len(trainset) * self.split_ratio)
        train_val_split = [train_dataset_length, len(trainset) - train_dataset_length]

        self.data_train, self.data_val = random_split(trainset, train_val_split)
        self.data_test = CIFAR10(self.data_dir, train=False, transform=transforms.ToTensor())

        self.input_dims = self.data_train[0][0].shape
        self.input_size = np.prod(self.input_dims)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size)
