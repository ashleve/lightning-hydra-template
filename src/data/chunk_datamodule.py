from typing import Any, Dict, Optional, Tuple, List, Iterator, Union
import numpy as np
import pandas as pd
import sqlite3
import math
import os

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Sampler 
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch import Tensor

from torch import default_generator, randperm



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
        event_no = idx #self.event_nos[idx]
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

class ChunkIceCubeSQLDatamodule(LightningDataModule):
    def __init__(
        self,
        db_path: str,
        pulsemap: str,
        input_cols: List[str],
        target_cols: str,
        chunk_csv_train: List[str],
        chunk_csv_test: List[str],
        chunk_csv_val: List[str],
        batch_sizes: List[int],
        truth_table: str = "truth",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if not self.data_train and not self.data_val and not self.data_test:
            # csv_filenames_train = [os.path.join(self.hparams.csv_folder, "train", f"output_{i}.csv") for i in range(1, 8)]
            # csv_filenames_val = [os.path.join(self.hparams.csv_folder, "val", f"output_{i}.csv") for i in range(1, 8)]
            # csv_filenames_test = [os.path.join(self.hparams.csv_folder, "test", f"output_{i}.csv") for i in range(1, 8)]
            self.data_train = ChunkDataset(
                db_path=self.hparams.db_path,
                chunk_csvs=self.hparams.chunk_csv_train,
                pulsemap=self.hparams.pulsemap,
                truth_table=self.hparams.truth_table,
                target_cols=self.hparams.target_cols,
                input_cols=self.hparams.input_cols
            )
            self.data_val = ChunkDataset(
                db_path=self.hparams.db_path,
                chunk_csvs=self.hparams.chunk_csv_val,
                pulsemap=self.hparams.pulsemap,
                truth_table=self.hparams.truth_table,
                target_cols=self.hparams.target_cols,
                input_cols=self.hparams.input_cols
            )
            self.data_test = ChunkDataset(
                db_path=self.hparams.db_path,
                chunk_csvs=self.hparams.chunk_csv_test,
                pulsemap=self.hparams.pulsemap,
                truth_table=self.hparams.truth_table,
                target_cols=self.hparams.target_cols,
                input_cols=self.hparams.input_cols
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn,
            batch_sampler=ChunkSampler(chunk_csvs=self.hparams.chunk_csv_train, batch_sizes=self.hparams.batch_sizes) 
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn,
            batch_sampler=ChunkSampler(chunk_csvs=self.hparams.chunk_csv_val, batch_sizes=self.hparams.batch_sizes) 
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn,
            batch_sampler=ChunkSampler(chunk_csvs = self.hparams.chunk_csv_test, batch_sizes=self.hparams.batch_sizes)
        )

    def teardown(self, stage: Optional[str] = None):
        # self.data_test.close_connection()
        # self.data_val.close_connection()
        # self.data_train.close_connection()
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        pass




if __name__ == "__main__":
    _ = ChunkIceCubeSQLDatamodule()