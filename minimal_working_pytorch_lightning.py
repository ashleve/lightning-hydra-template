from typing import Any, Dict, Optional, Tuple, List, Callable, Union
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
from lightning import LightningDataModule, LightningModule, Trainer
from torchdata.datapipes.iter import IterableWrapper, IterDataPipe, ShardingFilter, Shuffler, Mapper, RandomSplitter
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService

def test_func(x):
    data, label = x
    data = data*2 
    return (data, label)

class TestDataModule(LightningDataModule):
    def __init__(self, num_workers):
        super().__init__()
        self.num_workers = num_workers
        self.data_size = 2000
        self.feature_size = 100

        # data = torch.randn(self.data_size, self.feature_size)
        # labels = torch.randint(0, 2, (self.data_size,))

        

        # self.train_datapipe = IterableWrapper(zip(data,labels)) \
        #     .shuffle() \
        #     .sharding_filter() \
        #     .map(test_func)
        # self.test_datapipe = IterableWrapper(zip(data,labels)) \
        #     .shuffle() \
        #     .sharding_filter() \
        #     .map(test_func)
        # self.val_datapipe = IterableWrapper(zip(data,labels)) \
        #     .shuffle() \
        #     .sharding_filter() \
        #     .map(test_func)
        
        # self.rs = MultiProcessingReadingService(num_workers=self.num_workers)

        # self.val_dataloader2 = DataLoader2(
        #     self.val_datapipe,
        #     reading_service=self.rs,
        # )
        # self.train_dataloader2 = DataLoader2(
        #     self.train_datapipe,
        #     reading_service=self.rs,
        # )
        # self.test_dataloader2 = DataLoader2(
        #     self.test_datapipe,
        #     reading_service=self.rs,
        # )
        self.train_datapipe: Optional[IterDataPipe] = None
        self.val_datapipe: Optional[IterDataPipe] = None
        self.test_datapipe: Optional[IterDataPipe] = None

        self.rs: Optional[MultiProcessingReadingService] = None

        self.train_dataloader2: Optional[DataLoader2] = None
        self.val_dataloader2: Optional[DataLoader2] = None
        self.test_dataloader2: Optional[DataLoader2] = None

    def prepare_data(self):
        # data = torch.randn(self.data_size, self.feature_size)
        # labels = torch.randint(0, 2, (self.data_size,))

        # self.train_datapipe = IterableWrapper(zip(data,labels)) \
        #     .shuffle() \
        #     .sharding_filter() \
        #     .map(test_func)
        # self.test_datapipe = IterableWrapper(zip(data,labels)) \
        #     .shuffle() \
        #     .sharding_filter() \
        #     .map(test_func)
        # self.val_datapipe = IterableWrapper(zip(data,labels)) \
        #     .shuffle() \
        #     .sharding_filter() \
        #     .map(test_func)
        pass

    def setup(self, stage: Optional[str] = None):
        # Define the datapipes for train, val, test
        if not self.train_datapipe or not self.val_datapipe or not self.test_datapipe:
            data = torch.randn(self.data_size, self.feature_size)
            labels = torch.randint(0, 2, (self.data_size,))

            

            self.train_datapipe = IterableWrapper(zip(data,labels)) \
                .shuffle() \
                .sharding_filter() \
                .map(test_func)
            self.test_datapipe = IterableWrapper(zip(data,labels)) \
                .shuffle() \
                .sharding_filter() \
                .map(test_func)
            self.val_datapipe = IterableWrapper(zip(data,labels)) \
                .shuffle() \
                .sharding_filter() \
                .map(test_func)
        if not self.rs:
            self.rs = MultiProcessingReadingService(num_workers=self.num_workers)

        if not self.train_dataloader2 or not self.val_dataloader2 or not self.test_dataloader2:
            self.val_dataloader2 = DataLoader2(
                self.val_datapipe,
                reading_service=self.rs,
            )
            self.train_dataloader2 = DataLoader2(
                self.train_datapipe,
                reading_service=self.rs,
            )
            self.test_dataloader2 = DataLoader2(
                self.test_datapipe,
                reading_service=self.rs,
            )
        # pass
        # self.rs = MultiProcessingReadingService(num_workers=self.num_workers)


    def train_dataloader(self):
        # rs = MultiProcessingReadingService(num_workers=self.num_workers)
        # self.train_dataloader = DataLoader2(
        #     self.train_datapipe,
        #     reading_service=rs,
        # )
        return self.train_dataloader2
        # return DataLoader2(
        #     self.train_datapipe,
        #     reading_service=rs,
        # )

    def val_dataloader(self):
        
        return self.val_dataloader2
        
        # return DataLoader2(
        #     self.val_datapipe, 
        #     reading_service=rs,
        # )

    def test_dataloader(self):
        # rs = MultiProcessingReadingService(num_workers=self.num_workers)
        # self.test_dataloader = DataLoader2(
        #     self.test_datapipe,
        #     reading_service=rs,
        # )
        return self.test_dataloader2
        # return DataLoader2(
        #     self.test_datapipe,
        #     reading_service=rs,
        # )

# Define the model
class SimpleNN(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(100, 1)

    def forward(self, x):
        return self.layer(x).squeeze()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.BCEWithLogitsLoss()(y_hat, y.float())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.BCEWithLogitsLoss()(y_hat, y.float())
        acc = ((y_hat > 0).float() == y.float()).float().mean()
        metrics = {"val_loss": loss, "val_acc": acc}
        self.log_dict(metrics)
        return metrics

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)

# Training the model
model = SimpleNN()
dm = TestDataModule( num_workers=2)
trainer = Trainer(accelerator="gpu", devices=1, min_epochs=1, max_epochs=3, precision=16)
trainer.fit(model, dm)
dm.val_dataloader2.shutdown()
dm.test_dataloader2.shutdown()
dm.train_dataloader2.shutdown()
# pkill -9 -f "minimal_working_pytorch_lightning.py"