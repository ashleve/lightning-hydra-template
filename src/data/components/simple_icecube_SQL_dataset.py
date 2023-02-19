from torch.utils.data import Dataset
from torch import Tensor
import sqlite3

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