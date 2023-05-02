from typing import Any, Dict, Optional, Tuple, List, Callable, Union
import numpy as np

import pandas as pd
import torch
import torch.nn as nn
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, SequentialSampler #random_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import Dataset, DataLoader, random_split
from torch import Tensor
import sqlite3
import math
from torch import default_generator, randperm
from torch._utils import _accumulate
from torch.utils.data.dataset import Subset

from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService
import torchdata.datapipes as dp
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterableWrapper, IterDataPipe, Mapper, MaxTokenBucketizer, ShardingFilter


@functional_datapipe("read_csv")
class ReadCSV(IterDataPipe):
    def __init__(self, csv_file):
        self.csv_file = csv_file

    def __iter__(self):
        with open(self.csv_file, "r") as f:
            for line in f:
                yield int(line.strip())

@functional_datapipe("read_csv_dp")
class ReadCSVMultiple(IterDataPipe):
    def __init__(self, datapipe):
        self.datapipe = datapipe

    def __iter__(self):
        for csv_file_path in self.datapipe:
            with open(csv_file_path, "r") as f:
                for line in f:
                    yield int(line.strip())

@functional_datapipe("query_sql")
class QuerySQL(IterDataPipe):
    def __init__(self, datapipe, db_path, input_cols, pulsemap, target_cols, truth_table):

        self.datapipe = datapipe
        self.db_path = db_path
        self.input_cols_str = ", ".join(input_cols)
        self.target_cols_str = ", ".join(target_cols)
        self.pulsemap = pulsemap
        self.truth_table = truth_table

    def __iter__(self):
        with sqlite3.connect(self.db_path) as conn:
            for event_no in self.datapipe:
                features = torch.Tensor(conn.execute(f"SELECT {self.input_cols_str} FROM {self.pulsemap} WHERE event_no == {event_no}").fetchall())
                truth = torch.Tensor(conn.execute(f"SELECT {self.target_cols_str} FROM {self.truth_table} WHERE event_no == {event_no}").fetchall())
                yield (features, truth)
def upgrade_transform_func(x):
    features, truth = x
    features[:, 0] = torch.log10(features[:, 0]) / 2.0  # charge
    features[:, 1] /= 2e04  # dom_time
    features[:, 1] -= 1.0
    features[:, 2] /= 500.0  # dom_x
    features[:, 3] /= 500.0  # dom_y
    features[:, 4] /= 500.0  # dom_z
    features[:, 5] /= 0.05  # pmt_area
    # features[:,6] /= 1.  # pmt_dir_x
    # features[:,7] /= 1.  # pmt_dir_y
    # features[:,8] /= 1.  # pmt_dir_z
    truth = torch.log10(truth)
    return (features, truth)

@functional_datapipe("transform_data")
class TransfromData(IterDataPipe):
    def __init__(self, datapipe, feature_transform, truth_transform = None):
        self.datapipe = datapipe 
        # self.input_cols = input_cols
        # self.target_cols = target_cols
        self.feature_transform = feature_transform

        if not truth_transform:
            self.truth_transform = lambda features : features
        else:
          self.truth_transform = truth_transform


    def __iter__(self):
        for features, truth in self.datapipe:
            features = self.feature_transform(features)
            truth = self.truth_transform(truth)

            yield (features, truth)


def upgrade_feature_transform(features):
    features[:, 0] = torch.log10(features[:, 0]) / 2.0  # charge
    features[:, 1] /= 2e04  # dom_time
    features[:, 1] -= 1.0
    features[:, 2] /= 500.0  # dom_x
    features[:, 3] /= 500.0  # dom_y
    features[:, 4] /= 500.0  # dom_z
    features[:, 5] /= 0.05  # pmt_area
    # features[:,6] /= 1.  # pmt_dir_x
    # features[:,7] /= 1.  # pmt_dir_y
    # features[:,8] /= 1.  # pmt_dir_z
    return features
    

def Prometheus_feature_transform(features):
    features[:, 0] /= 100.0  # dom_x
    features[:, 1] /= 100.0  # dom_y
    features[:, 2] += 350.0  # dom_z
    features[:, 2] /= 100.0
    features[:, 3] /= 1.05e04  # dom_time
    features[:, 3] -= 1.0
    features[:, 3] *= 20.0
    return features

def log10_target_transform(target):
   return torch.log10(target)


@functional_datapipe("pad_batch")
class PadBatch(IterDataPipe):
    def __init__(self, batch):
        self.batch = batch
        
    def __iter__(self):
        for batch in self.batch:

          (xx, y) = zip(*batch)
          x_lens = [len(x) for x in xx]
          xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)

          pad_mask = torch.zeros_like(xx_pad[:, :, 0]).type(torch.bool)

          for i, length in enumerate(x_lens):
              pad_mask[i, length:] = True

          yield (xx_pad, torch.tensor(y), pad_mask)


def len_fn(datapipe):
  features, _ = datapipe
  return features.shape[0]
    
def make_datapipe(
        csv_file, 
        db_path, 
        input_cols, 
        pulsemap, 
        target_cols, 
        truth_table, 
        max_token_count,
        feature_transform,
        truth_transform = None
    ):
    datapipe = ReadCSV( csv_file).sharding_filter() \
        .query_sql(
        db_path = db_path, 
        input_cols = input_cols, 
        pulsemap = pulsemap, 
        target_cols = target_cols,
        truth_table = truth_table,
        ) \
        .map(upgrade_transform_func
        ) \
        .max_token_bucketize(
        max_token_count = max_token_count,
        len_fn = len_fn,
        include_padding = True
        ) \
        .pad_batch()
    return datapipe

def make_train_test_val_datapipe(
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
    ):
    train_datapipe = make_datapipe(
        train_csv_file,
        db_path,
        input_cols,
        pulsemap,
        target_cols,
        truth_table,
        max_token_count,
        feature_transform,
        truth_transform
    )
    test_datapipe = make_datapipe(
        test_csv_file,
        db_path,
        input_cols,
        pulsemap,
        target_cols,
        truth_table,
        max_token_count,
        feature_transform,
        truth_transform
    )
    val_datapipe = make_datapipe(
        val_csv_file,
        db_path,
        input_cols,
        pulsemap,
        target_cols,
        truth_table,
        max_token_count,
        feature_transform,
        truth_transform
    )
    return train_datapipe, test_datapipe, val_datapipe


class IceCubeDatamodule(LightningDataModule):
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
        max_token_count: int = 64,
        num_workers: int = 16,
        multi_processing_reading_service_num_workers: int = 4,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations here if any

        self.datapipe_train: Optional[IterDataPipe] = None
        self.datapipe_val: Optional[IterDataPipe] = None
        self.datapipe_test: Optional[IterDataPipe] = None

        self.dataloader_train: Optional[DataLoader2] = None
        self.dataloader_val: Optional[DataLoader2] = None
        self.dataloader_test: Optional[DataLoader2] = None

        self.rs = MultiProcessingReadingService(
            num_workers = self.hparams.multi_processing_reading_service_num_workers
            )

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
        if not self.datapipe_train and not self.datapipe_val and not self.datapipe_test:
            self.datapipe_train, self.datapipe_val, self.datapipe_test = make_train_test_val_datapipe(
                train_csv_file = self.hparams.train_csv_file,
                test_csv_file = self.hparams.test_csv_file,
                val_csv_file = self.hparams.val_csv_file,
                db_path = self.hparams.db_path, 
                input_cols = self.hparams.input_cols, 
                pulsemap = self.hparams.pulsemap, 
                target_cols = self.hparams.target_cols, 
                truth_table = self.hparams.truth_table, 
                max_token_count = self.hparams.max_token_count,
                feature_transform = upgrade_feature_transform,
                truth_transform = None,
            )
        if not self.dataloader_train and not self.dataloader_val and not self.dataloader_test:
            self.dataloader_train = DataLoader2(
            datapipe = self.datapipe_train,
            reading_service = self.rs,
            )
            self.dataloader_val = DataLoader2( 
            datapipe = self.datapipe_val,
            reading_service = self.rs,
            )
            self.dataloader_test = DataLoader2( 
            datapipe = self.datapipe_test,
            reading_service = self.rs,
            )

    def train_dataloader(self):
        
        return self.dataloader_train

    def val_dataloader(self):
        
        return self.dataloader_val

    def test_dataloader(self):
        return self.dataloader_test
            
    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        self.rs.finalize()
        # self.dataloader_train.shutdown()
        # self.dataloader_test.shutdown()
        # self.dataloader_val.shutdown()
        # pass
    def dataloader_shutdown(self):
        self.dataloader_train.shutdown()
        self.dataloader_test.shutdown()
        self.dataloader_val.shutdown()

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    _ = IceCubeDatamodule()
