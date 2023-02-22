from typing import Any, Dict, Optional, Tuple, List
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset #random_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch import Tensor
import sqlite3
import math
from torch import default_generator, randperm
from torch._utils import _accumulate
from torch.utils.data.dataset import Subset


def random_split(dataset, lengths,
                 generator=default_generator):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.

    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))
    >>> random_split(range(30), [0.3, 0.3, 0.4], generator=torch.Generator(
    ...   ).manual_seed(42))

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: List[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(f"Length of split at index {i} is 0. "
                              f"This might result in an empty dataset.")

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):    # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[call-overload]
    return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]

# def pad_collate(batch):
#   (xx, y) = zip(*batch)
#   x_lens = [len(x) for x in xx]
  
#   xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)

#   return xx_pad, y, x_lens
def pad_collate(batch):
  (xx, y) = zip(*batch)
  x_lens = [len(x) for x in xx]
  
#   print([x.shape for x in xx])
  xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)#float("inf")
  # yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)

  pad_mask = torch.zeros_like(xx_pad[:, :, 0]).type(torch.bool)
  for i, length in enumerate(x_lens):
    pad_mask[i, length:] = True

  return xx_pad, torch.tensor(y), pad_mask

class SimpleDataset(Dataset):
  def __init__(self, 
               db_path: str, 
               event_no_list_path: str,
               pulsemap: str,
               input_cols: List[str],
               target_cols: List[str],
               truth_table: str = "truth"
               ):
    self.db_path = db_path
    self.event_no_list_path = event_no_list_path
    self.pulsemap = pulsemap
    self.input_cols = input_cols
    self.target_cols = target_cols
    self.truth_table = truth_table


    if isinstance(list(input_cols), list):
      self.input_cols_str = ", ".join(input_cols)
    else:

      self.input_cols_str = input_cols

    if isinstance(target_cols, list):
      self.target_cols_str = ", ".join(target_cols)
    else:
      self.target_cols_str = target_cols
    
    self.event_no_list = np.genfromtxt(self.event_no_list_path,dtype=int)

    self.data_len = len(self.event_no_list)
    

  def __getitem__(self, index):
    event_no = self.event_no_list[index]
    # print(event_no)
    with sqlite3.connect(self.db_path) as conn:
      features = Tensor(conn.execute(f"SELECT {self.input_cols_str} FROM {self.pulsemap} WHERE event_no == {event_no}").fetchall())
    #   print(conn.execute(f"SELECT pid FROM {self.truth_table} WHERE event_no == {event_no}").fetchall())
      truth = Tensor(conn.execute(f"SELECT {self.target_cols_str} FROM {self.truth_table} WHERE event_no == {event_no}").fetchall())
    #   print(features.shape)
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
        pulsemap: str,
        input_cols: List[str],
        target_cols: List[str],
        truth_table: str = "truth",
        data_dir: str = "data/",
        train_val_test_split: Tuple[float, float, float] = (0.8, 0.1, 0.1),# train_val_test_split_rate: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        # print()
        # print(train_val_test_split)
        # print(sum(train_val_test_split))
        # print()

        # event_no_list: List[int],

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
        
        dataset = SimpleDataset(
            db_path = self.hparams.db_path, 
            event_no_list_path = self.hparams.event_no_list_path,
            pulsemap = self.hparams.pulsemap,
            input_cols = self.hparams.input_cols,
            target_cols = self.hparams.target_cols,
            truth_table = self.hparams.truth_table,
        )
        # lengths = [int(p * len(dataset)) for p in self.hparams.train_val_test_split]
        # lengths[-1] = len(dataset) - sum(lengths[:-1])
        # print()
        # print(lengths)
        # print()

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            
            # delete when upgrading torch
            

            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths= self.hparams.train_val_test_split, # revert when upgrading torch
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
