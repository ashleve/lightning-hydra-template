from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule
from torchvision.transforms import transforms
from torchvision.datasets import MNIST


class MNISTDataModule(LightningDataModule):
    """
        This is example of datamodule for MNIST digits dataset.
        All data modules should be located in separate folders with file named 'datamodule.py' containing class which
        is always called 'DataModule'!

        The folder name of datamodule used during training should be specified in run config and all parameters from
        'dataset' section will be passed in 'hparams' dictionary.

        All datamodules should have a structure like this one!
    """
    def __init__(self, hparams):
        super().__init__()

        # hparams["data_dir"] is always automatically set to 'data_path' from 'project_config.yaml'
        self.data_dir = hparams["data_dir"]

        self.batch_size = hparams.get("batch_size") or 64
        self.train_val_split_ratio = hparams.get("train_val_split_ratio") or 0.9
        self.train_val_split = hparams.get("train_val_split") or None
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

        if not self.train_val_split:
            train_length = int(len(trainset) * self.train_val_split_ratio)
            val_length = len(trainset) - train_length
            self.train_val_split = [train_length, val_length]

        self.data_train, self.data_val = random_split(trainset, self.train_val_split)
        self.data_test = MNIST(self.data_dir, train=False, transform=self.transforms)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)
