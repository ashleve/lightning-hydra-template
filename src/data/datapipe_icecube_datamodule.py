from typing import Any, Dict, Optional, Tuple, List
import numpy as np

import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, SequentialSampler #random_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch import Tensor
import sqlite3
import math
from torch import default_generator, randperm
from torch._utils import _accumulate
from torch.utils.data.dataset import Subset



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
        pulsemap: str,
        train_csv_file: str, 
        test_csv_file: str,
        val_csv_file: str,
        input_cols: List[str],
        target_cols: List[str],
        truth_table: str = "truth",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations here if any


        # self.event_no_list = np.genfromtxt(self.hparams.event_no_list_path,dtype=int)
    
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    # @property
    # def num_classes(self):
    #     return 10

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # sampler = SequentialSampler()
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train, self.data_val, self.data_test = make_train_test_val_datapipe(
                train_csv_file,
                test_csv_file,
                val_csv_file,
                db_path, 
                input_cols, 
                pulsemap, 
                target_cols, 
                truth_table, 
                max_token_count,
                feature_transform,
                truth_transform = None
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
