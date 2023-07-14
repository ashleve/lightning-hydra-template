from typing import Any, Dict, Optional, Tuple, List, Iterator, Union
import numpy as np

import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler, SequentialSampler #random_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch import Tensor
import sqlite3

chunk_csv_train = [
  "/groups/icecube/moust/storage/cached_event_no/upgrade_numu/train/output_1.csv",
  "/groups/icecube/moust/storage/cached_event_no/upgrade_numu/train/output_2.csv",
  "/groups/icecube/moust/storage/cached_event_no/upgrade_numu/train/output_3.csv",
  "/groups/icecube/moust/storage/cached_event_no/upgrade_numu/train/output_4.csv",
  "/groups/icecube/moust/storage/cached_event_no/upgrade_numu/train/output_5.csv",
  "/groups/icecube/moust/storage/cached_event_no/upgrade_numu/train/output_6.csv",
  "/groups/icecube/moust/storage/cached_event_no/upgrade_numu/train/output_7.csv",
]
chunk_csv_test = [
  "/groups/icecube/moust/storage/cached_event_no/upgrade_numu/test/output_1.csv",
  "/groups/icecube/moust/storage/cached_event_no/upgrade_numu/test/output_2.csv",
  "/groups/icecube/moust/storage/cached_event_no/upgrade_numu/test/output_3.csv",
  "/groups/icecube/moust/storage/cached_event_no/upgrade_numu/test/output_4.csv",
  "/groups/icecube/moust/storage/cached_event_no/upgrade_numu/test/output_5.csv",
  "/groups/icecube/moust/storage/cached_event_no/upgrade_numu/test/output_6.csv",
  "/groups/icecube/moust/storage/cached_event_no/upgrade_numu/test/output_7.csv",
]
chunk_csv_val = [
  "/groups/icecube/moust/storage/cached_event_no/upgrade_numu/val/output_1.csv",
  "/groups/icecube/moust/storage/cached_event_no/upgrade_numu/val/output_2.csv",
  "/groups/icecube/moust/storage/cached_event_no/upgrade_numu/val/output_3.csv",
  "/groups/icecube/moust/storage/cached_event_no/upgrade_numu/val/output_4.csv",
  "/groups/icecube/moust/storage/cached_event_no/upgrade_numu/val/output_5.csv",
  "/groups/icecube/moust/storage/cached_event_no/upgrade_numu/val/output_6.csv",
  "/groups/icecube/moust/storage/cached_event_no/upgrade_numu/val/output_7.csv",
]

batch_sizes = [512, 256, 128, 64, 32, 16, 8]
truth_table = "truth"
db_path = "/groups/icecube/petersen/GraphNetDatabaseRepository/Upgrade_Data/sqlite3/dev_step4_upgrade_028_with_noise_dynedge_pulsemap_v3_merger_aftercrash.db"
pulsemap = "SplitInIcePulses_dynedge_v2_Pulses"
input_cols =  ["dom_x", "dom_y", "dom_z", "dom_time", "charge"]
target_cols = "inelasticity"

class ChunkDataset(Dataset):
    """
    PyTorch dataset for loading chunked data from an SQLite database.
    This dataset retrieves pulsemap and truth data for each event from the database.

    Args:
        db_filename (str): Filename of the SQLite database.
        csv_filenames (list of str): List of CSV filenames containing event numbers.
        pulsemap_table (str): Name of the table containing pulsemap data.
        truth_table (str): Name of the table containing truth data.
        truth_variable (str): Name of the variable to query from the truth table.
        feature_variables (list of str): List of variable names to query from the pulsemap table.
    """

    def __init__(
        self,
        db_path: str,
        chunk_csvs: List[str],
        pulsemap: str,
        truth_table: str,
        target_cols: str,
        input_cols: List[str]
    ) -> None:
        self.conn = sqlite3.connect(db_path)  # Connect to the SQLite database
        self.c = self.conn.cursor()
        self.event_nos = []
        for csv_filename in chunk_csvs:
            df = pd.read_csv(csv_filename)
            self.event_nos.extend(df['event_no'].tolist())  # Collect event numbers from CSV files
        self.pulsemap = pulsemap  # Name of the table containing pulsemap data
        self.truth_table = truth_table  # Name of the table containing truth data
        self.target_cols = target_cols  # Name of the variable to query from the truth table
        self.input_cols = input_cols  # List of variable names to query from the pulsemap table


    def __len__(self) -> int:
        return len(self.event_nos)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # print(idx)
        event_no = idx # self.event_nos[idx]
        # print(event_no)
        # Query the truth variable for the given event number
        self.c.execute(f"SELECT {self.target_cols} FROM {self.truth_table} WHERE event_no = ?", (event_no,))
        truth_value = self.c.fetchone()[0]
        input_query = ', '.join(self.input_cols)
        # Query the feature variables from the pulsemap table for the given event number
        self.c.execute(f"SELECT {input_query} FROM {self.pulsemap} WHERE event_no = ?", (event_no,))
        pulsemap_data = self.c.fetchall()
        return torch.tensor(truth_value, dtype=torch.float32), torch.tensor(pulsemap_data, dtype=torch.float32)
    
    def close_connection(self) -> None:
        self.conn.close()


class ChunkSampler(Sampler):
    """
    PyTorch sampler for creating chunks from event numbers.

    Args:
        csv_filenames (List[str]): List of CSV filenames containing event numbers.
        batch_sizes (List[int]): List of batch sizes for each CSV file.
    """

    def __init__(
        self, 
        chunk_csvs: List[str], 
        batch_sizes: List[int]
    ) -> None:
        self.event_nos = []
        for csv_filename, batch_size in zip(chunk_csvs, batch_sizes):
            event_nos = pd.read_csv(csv_filename)['event_no'].tolist()
            self.event_nos.extend([event_nos[i:i + batch_size] for i in range(0, len(event_nos), batch_size)])

    def __iter__(self) -> Iterator:
        return iter(self.event_nos)

    def __len__(self) -> int:
        return len(self.event_nos)
    
def collate_fn(data: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Split the data into truths and pulsemap data
    truths, pulsemap_data = zip(*data)

    # Pad the pulsemap data
    pulsemap_lengths = [len(x) for x in pulsemap_data]
    pulsemap_data = pad_sequence(pulsemap_data, batch_first=True, padding_value=0)

    pad_mask = torch.zeros_like(pulsemap_data[:, :, 0]).type(torch.bool)

    for i, length in enumerate(pulsemap_lengths):
        pad_mask[i, length:] = True

    return (pulsemap_data, torch.stack(truths), pad_mask)
    
dataset = ChunkDataset(
    db_path=db_path, 
    chunk_csvs=chunk_csv_train, 
    pulsemap=pulsemap, 
    truth_table=truth_table, 
    target_cols=target_cols, 
    input_cols=input_cols
    )

dl = DataLoader(
    dataset=dataset,
    collate_fn=collate_fn,
    batch_sampler=ChunkSampler(chunk_csv_train, batch_sizes),
    num_workers=12,
    )

for i, (features, _, _) in enumerate(dl):
    print(i)
    print(features.shape)
    