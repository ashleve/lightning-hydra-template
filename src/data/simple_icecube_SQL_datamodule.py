from typing import Any, Dict, Optional, Tuple, List
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch import Tensor
import sqlite3

def pad_collate(batch):
  (xx, y) = zip(*batch)
  x_lens = [len(x) for x in xx]
  
  xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)

  return xx_pad, y, x_lens

class SimpleDataset(Dataset):
  def __init__(self, 
               db_path, 
               event_no_list,
               pulsemap,
               input_cols,
               target_cols,
               truth_table = "truth"
               ):
    self.db_path = db_path
    self.event_no_list = event_no_list
    self.pulsemap = pulsemap
    self.input_cols = input_cols
    self.target_cols = target_cols
    self.truth_table = truth_table

    if isinstance(input_cols, list):
      self.input_cols_str = ", ".join(input_cols)
    else:
      self.input_cols_str = input_cols

    if isinstance(target_cols, list):
      self.target_cols_str = ", ".join(target_cols)
    else:
      self.target_cols_str = target_cols
      
    self.data_len = len(event_no_list)
    

  def __getitem__(self, index):
    event_no = self.event_no_list[index]
    with sqlite3.connect(self.db_path) as conn:
      features = Tensor(conn.execute(f"SELECT {self.input_cols_str} FROM {self.pulsemap} WHERE event_no == {event_no}").fetchall())
      truth = Tensor(conn.execute(f"SELECT {self.target_cols_str} FROM {self.truth_table} WHERE event_no == {event_no}").fetchall())

    return features, truth
  
  def __len__(self):
    return self.data_len

class SimpleIceCubeSQLDatamodule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self,
        db_path: str,
        event_no_list_path: str,
        event_no_list: List[int],
        pulsemap: str,
        input_cols: List[str],
        target_cols: List[str],
        truth_table: str = "truth",
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),# train_val_test_split_rate: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # assert sum(train_val_test_split_rate) <= 1

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self):
        return 10

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass
        # MNIST(self.hparams.data_dir, train=True, download=True)
        # MNIST(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        event_no_list = np.genfromtxt(self.hparams.event_no_list_path,dtype=int)
        dataset = SimpleDataset(
            db_path = self.hparams.db_path, 
            event_no_list = event_no_list,
            pulsemap = self.hparams.pulsemap,
            input_cols = self.hparams.input_cols,
            target_cols = self.hparams.target_cols,
            truth_table = self.hparams.truth_table, 
               
        )
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn= pad_collate,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn= pad_collate,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn= pad_collate,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    _ = SimpleIceCubeSQLDatamodule()
