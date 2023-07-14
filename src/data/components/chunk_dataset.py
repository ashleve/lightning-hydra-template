import sqlite3
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler
from typing import List, Tuple, Iterator

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
        db_filename: str,
        csv_filenames: List[str],
        pulsemap_table: str,
        truth_table: str,
        truth_variable: str,
        feature_variables: List[str]
    ) -> None:
        self.conn = sqlite3.connect(db_filename)  # Connect to the SQLite database
        self.c = self.conn.cursor()
        self.event_nos = []
        for csv_filename in csv_filenames:
            df = pd.read_csv(csv_filename)
            self.event_nos.extend(df['event_no'].tolist())  # Collect event numbers from CSV files
        self.pulsemap_table = pulsemap_table  # Name of the table containing pulsemap data
        self.truth_table = truth_table  # Name of the table containing truth data
        self.truth_variable = truth_variable  # Name of the variable to query from the truth table
        self.feature_variables = feature_variables  # List of variable names to query from the pulsemap table

    def __len__(self) -> int:
        return len(self.event_nos)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        event_no = self.event_nos[idx]
        # Query the truth variable for the given event number
        self.c.execute(f"SELECT {self.truth_variable} FROM {self.truth_table} WHERE event_no = ?", (event_no,))
        truth_value = self.c.fetchone()[0]
        feature_query = ', '.join(self.feature_variables)
        # Query the feature variables from the pulsemap table for the given event number
        self.c.execute(f"SELECT {feature_query} FROM {self.pulsemap_table} WHERE event_no = ?", (event_no,))
        pulsemap_data = self.c.fetchall()
        return torch.tensor(truth_value, dtype=torch.float32), torch.tensor(pulsemap_data, dtype=torch.float32)


class ChunkSampler(Sampler):
    """
    PyTorch sampler for creating chunks from event numbers.

    Args:
        csv_filenames (List[str]): List of CSV filenames containing event numbers.
        batch_sizes (List[int]): List of batch sizes for each CSV file.
    """

    def __init__(self, csv_filenames: List[str], batch_sizes: List[int]) -> None:
        self.event_nos = []
        for csv_filename, batch_size in zip(csv_filenames, batch_sizes):
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



# class ChunkDataset(Dataset):
#     def __init__(self, db_filename, csv_filenames, pulsemap_table, truth_table):
#         # self.conn = sqlite3.connect(db_filename)
#         # self.c = self.conn.cursor()
#         # self.c.execute("SELECT event_no FROM truth")
#         # self.event_nos = [row[0] for row in self.c.fetchall()]
#         self.conn = sqlite3.connect(db_filename)
#         self.c = self.conn.cursor()
#         self.event_nos = []
#         for csv_filename in csv_filenames:
#             df = pd.read_csv(csv_filename)
#             self.event_nos.extend(df['event_no'].tolist())

#     def __len__(self):
#         return len(self.event_nos)

#     def __getitem__(self, idx):
#         event_no = self.event_nos[idx]
#         self.c.execute("SELECT energy FROM truth WHERE event_no = ?", (event_no,))
#         energy = self.c.fetchone()[0]
#         self.c.execute("SELECT dom_x, dom_y, dom_z, dom_time, charge FROM SplitInIcePulses_dynedge_v2_Pulses WHERE event_no = ?", (event_no,))
#         pulsemap_data = self.c.fetchall()
#         return torch.tensor(energy, dtype=torch.float32), torch.tensor(pulsemap_data, dtype=torch.float32)
    
# class ChunkSampler(Sampler):
#     def __init__(self, csv_filenames, batch_sizes):
#         self.event_nos = []
#         for csv_filename, batch_size in zip(csv_filenames, batch_sizes):
#             # event_nos = np.loadtxt(csv_filename, delimiter=",", dtype=int).tolist()
#             event_nos = pd.read_csv(csv_filename)['event_no'].tolist()
#             self.event_nos.extend([event_nos[i:i + batch_size] for i in range(0, len(event_nos), batch_size)])

#     def __iter__(self):
#         return iter(self.event_nos)

#     def __len__(self):
#         return len(self.event_nos)
    
# def collate_fn(data):
#     # Split the data into energies and pulsemap data
#     energies, pulsemap_data = zip(*data)
    
#     # Pad the pulsemap data
#     pulsemap_lengths = [len(x) for x in pulsemap_data]
#     pulsemap_data = pad_sequence(pulsemap_data, batch_first=True, padding_value=0)

#     pad_mask = torch.zeros_like(pulsemap_data[:, :, 0]).type(torch.bool)

#     for i, length in enumerate(pulsemap_lengths):
#         pad_mask[i, length:] = True
    
#     return (pulsemap_data, torch.stack(energies), pad_mask)