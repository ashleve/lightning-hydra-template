import sqlite3
import torch

from torch.utils.data import Dataset
from torch import Tensor
from torchdata.datapipes.iter import  IterDataPipe, MaxTokenBucketizer
from torch.nn.utils.rnn import pad_sequence

class ReadCSV(IterDataPipe):
    def __init__(self, csv_file):
        self.csv_file = csv_file

    def __iter__(self):
        with open(self.csv_file, "r") as f:
            for line in f:
                yield int(line.strip())

class QuerySQL(IterDataPipe):
    def __init__(self, db_path, event_nos, input_cols, pulsemap, target_cols, truth_table):
        self.db_path = db_path
        self.event_nos = event_nos
        self.input_cols_str = ", ".join(input_cols)
        self.target_cols_str = ", ".join(target_cols)
        self.pulsemap = pulsemap
        self.truth_table = truth_table

    def __iter__(self):
        with sqlite3.connect(self.db_path) as conn:
            for event_no in self.event_nos:
                features = Tensor(conn.execute(f"SELECT {self.input_cols_str} FROM {self.pulsemap} WHERE event_no == {event_no}").fetchall())
                truth = Tensor(conn.execute(f"SELECT {self.target_cols_str} FROM {self.truth_table} WHERE event_no == {event_no}").fetchall())
                yield (features, truth)

class TransfromData(IterDataPipe):
    def __init__(self, datapipe, feature_transform, truth_transform = None):
        self.datapipe = datapipe
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

def Prometheus_featuer_transform(features):
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