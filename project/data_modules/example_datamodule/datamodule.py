from torch.utils.data import DataLoader
import pytorch_lightning as pl


class DataModule(pl.LightningDataModule):
    """
        This is example of datamodule.
        The folder name of datamodule used during training should be specified in run config and all parameters from
        'dataset' section will be passed in 'hparams' dictionary.

        All data modules should be located in separate folders and this class should always be called 'DataModule'!
        All datamodules should have a structure like this one!!

        See 'mnist_digits_datamodule' for more proper example.
    """
    def __init__(self, hparams):
        super().__init__()

        self.data_dir = hparams.get("data_dir") or "data/example_data"
        self.batch_size = hparams.get("batch_size") or 64
        self.num_workers = hparams.get("num_workers") or 1
        self.pin_memory = hparams.get("pin_memory") or False

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
