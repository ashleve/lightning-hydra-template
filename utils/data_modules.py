# pytorch
from torch.utils.data import DataLoader, random_split

# pytorch lightning
import pytorch_lightning as pl

# torchvision
from torchvision.datasets import MNIST
from torchvision import transforms


class MNISTDataModule(pl.LightningDataModule):

    def __init__(self, data_dir='data/mnist', batch_size=256):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.ToTensor()

        self.data_train = None
        self.data_val = None
        self.data_test = None

    def prepare_data(self):
        # download data
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # we set up only relevant datasets when stage is specified (automatically set by Pytorch-Lightning)
        if stage == 'fit' or stage is None:
            mnist_train = MNIST(self.data_dir, train=True, transform=self.transform)
            self.data_train, self.data_val = random_split(mnist_train, [55000, 5000])

        if stage == 'test' or stage is None:
            self.data_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size)
