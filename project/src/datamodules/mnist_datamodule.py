from torch.utils.data import DataLoader, ConcatDataset, random_split
from pytorch_lightning import LightningDataModule
from torchvision.transforms import transforms
from torchvision.datasets import MNIST


class MNISTDataModule(LightningDataModule):
    """
    This is example of lightning datamodule for MNIST dataset.
    To learn how to create datamodules visit:
        https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.data_dir = kwargs["data_dir"]
        self.batch_size = kwargs["batch_size"]
        self.train_val_test_split = kwargs["train_val_test_split"]
        self.num_workers = kwargs["num_workers"]
        self.pin_memory = kwargs["pin_memory"]

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # self.dims is returned when you call datamodule.size()
        self.dims = (1, 28, 28)

        self.data_train = None
        self.data_val = None
        self.data_test = None

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        trainset = MNIST(self.data_dir, train=True, transform=self.transforms)
        testset = MNIST(self.data_dir, train=False, transform=self.transforms)
        dataset = ConcatDataset(datasets=[trainset, testset])
        self.data_train, self.data_val, self.data_test = random_split(dataset, self.train_val_test_split)

    def train_dataloader(self):
        return DataLoader(dataset=self.data_train, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.data_val, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, shuffle=False)

    def test_dataloader(self):
        return DataLoader(dataset=self.data_test, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, shuffle=False)
