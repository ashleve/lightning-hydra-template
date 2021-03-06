import warnings
import torch
from pytorch_lightning import LightningDataModule
from torchtext import data, datasets

warnings.filterwarnings('ignore')


class IMDBDataModule(LightningDataModule):
    """
    This is example IMDB text classification dataset.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.data_dir = kwargs["data_dir"]
        self.batch_size = kwargs["batch_size"]
        self.val_test_split_ratio = kwargs["val_test_split_ratio"]
        self.max_vocab_size = kwargs["max_vocab_size"]

        self.TEXT = data.Field(include_lengths=True)
        self.LABEL = data.LabelField(dtype=torch.float)

        self.pad_idx = None
        self.vocab_size = None
        self.data_train = None
        self.data_val = None
        self.data_test = None

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        datasets.IMDB.download(self.data_dir)

    def setup(self, stage=None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        train_data, self.data_test = datasets.IMDB.splits(self.TEXT, self.LABEL, root=self.data_dir)
        self.data_train, self.data_val = train_data.split(split_ratio=self.val_test_split_ratio)

        self.TEXT.build_vocab(self.data_train, max_size=self.max_vocab_size)
        self.LABEL.build_vocab(self.data_train)
        self.pad_idx = self.TEXT.vocab.stoi[self.TEXT.pad_token]
        self.vocab_size = len(self.TEXT.vocab)

    def train_dataloader(self):
        return data.BucketIterator(self.data_train,
                                   batch_size=self.batch_size,
                                   sort_within_batch=True)

    def val_dataloader(self):
        return data.BucketIterator(self.data_val,
                                   batch_size=self.batch_size,
                                   sort_within_batch=True,
                                   train=False)

    def test_dataloader(self):
        return data.BucketIterator(self.data_test,
                                   batch_size=self.batch_size,
                                   sort_within_batch=True,
                                   train=False)
