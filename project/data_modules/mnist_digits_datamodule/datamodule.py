from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms
from torchvision.datasets import MNIST
import pytorch_lightning as pl


class DataModule(pl.LightningDataModule):
    """The proper datamodule for mnist digits dataset."""
    def __init__(self, hparams):
        super().__init__()

        self.data_dir = hparams.get("data_dir") or "data/mnist"
        self.batch_size = hparams.get("batch_size") or 64
        self.train_val_split_ratio = hparams.get("train_val_split_ratio") or 0.9
        self.num_workers = hparams.get("num_workers") or 1
        self.pin_memory = hparams.get("pin_memory") or False

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
