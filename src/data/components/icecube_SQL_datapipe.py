import sqlite3
import pandas as pd
import numpy as np
import csv
import random

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam

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

# def make_train_test_val_datapipe(
#         csv_file, db_path, 
#         input_cols, 
#         pulsemap, 
#         target_cols, 
#         truth_table, 
#         max_token_count,
#         feature_transform,
#         truth_transform = None
# ):
#     datapipe = ReadCSV( csv_file)
#     datapipe = QuerySQL( db_path, datapipe, input_cols, pulsemap, target_cols, truth_table)
#     datapipe = TransfromData( datapipe, feature_transform, truth_transform)
#     datapipe = MaxTokenBucketizer( datapipe, max_token_count = max_token_count, len_fn = len_fn, include_padding = True)
#     datapipe = PadBatch(datapipe)
    
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
        .transform_data(
        feature_transform, 
        truth_transform
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