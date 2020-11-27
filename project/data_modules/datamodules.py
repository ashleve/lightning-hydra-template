from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import transforms
import pytorch_lightning as pl


class ExampleDataModule(pl.LightningDataModule):
    """All datamodules should look like this!"""
    def __init__(self, data_dir="data/example_data", batch_size=64, data_split_ratio=0.90, num_workers=1,
                 pin_memory=False):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.data_split_ratio = data_split_ratio
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.transforms = None

        self.data_train = None
        self.data_val = None
        self.data_test = None

    def prepare_data(self):
        """Download data if needed."""
        pass

    def setup(self, stage=None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        pass

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='data/mnist', batch_size=64, train_val_split_ratio=0.90, num_workers=1,
                 pin_memory=False):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_val_split_ratio = train_val_split_ratio
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.transforms = transforms.ToTensor()

        self.data_train = None
        self.data_val = None
        self.data_test = None

    def prepare_data(self):
        """Download data if needed."""
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        trainset = MNIST(self.data_dir, train=True, transform=self.transforms)

        train_dataset_length = int(len(trainset) * self.train_val_split_ratio)
        train_val_split = [train_dataset_length, len(trainset) - train_dataset_length]

        self.data_train, self.data_val = random_split(trainset, train_val_split)
        self.data_test = MNIST(self.data_dir, train=False, transform=self.transforms)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir="data/cifar10", batch_size=64, train_val_split_ratio=0.90, num_workers=1,
                 pin_memory=False):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_val_split_ratio = train_val_split_ratio
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.transforms = transforms.ToTensor()

        self.data_train = None
        self.data_val = None
        self.data_test = None

    def prepare_data(self):
        """Download data if needed."""
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        trainset = CIFAR10(self.data_dir, train=True, transform=self.transforms)

        train_dataset_length = int(len(trainset) * self.train_val_split_ratio)
        train_val_split = [train_dataset_length, len(trainset) - train_dataset_length]

        self.data_train, self.data_val = random_split(trainset, train_val_split)
        self.data_test = CIFAR10(self.data_dir, train=False, transform=self.transforms)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)
