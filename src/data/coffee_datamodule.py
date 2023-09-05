from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from .components.PTG_dataset import PTG_Dataset

class CoffeeDataModule(LightningDataModule):
    """`LightningDataModule` for the MNIST dataset.

    The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples.
    It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a
    fixed-size image. The original black and white images from NIST were size normalized to fit in a 20x20 pixel box
    while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing
    technique used by the normalization algorithm. the images were centered in a 28x28 image by computing the center of
    mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        num_classes: int,
        sample_rate: int,
        window_size: int,
        split: int,
        epoch_length: int,
        pin_memory: bool,
    ) -> None:
        """Initialize a `CoffeeDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = None

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of activity classes.
        """
        return self.hparams.num_classes

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            exp_data = self.hparams.data_dir


            vid_list_file = f"{exp_data}/splits/train_activity.split{self.hparams.split}.bundle"
            vid_list_file_val = f"{exp_data}/splits/val.split{self.hparams.split}.bundle"
            vid_list_file_tst = f"{exp_data}/splits/test.split{self.hparams.split}.bundle"

            features_path = f"{exp_data}/features/"
            gt_path = f"{exp_data}/groundTruth/"
            mapping_file = f"{exp_data}/mapping.txt"

            #####################
            # Get Action Names
            #####################
            file_ptr = open(mapping_file, "r")
            actions = file_ptr.read().split("\n")[:-1]
            file_ptr.close()
            actions_dict = dict()
            for a in actions:
                actions_dict[a.split()[1]] = int(a.split()[0])

            #####################
            # Get Video Names
            #####################
            # Load training vidoes
            with open(vid_list_file, "r") as train_f:
                train_videos = train_f.read().split("\n")[:-1]

            # Load validation vidoes
            with open(vid_list_file_val, "r") as val_f:
                val_videos = val_f.read().split("\n")[:-1]

            # Load test videos
            with open(vid_list_file_tst, "r") as test_f:
                test_videos = test_f.read().split("\n")[:-1]


            self.data_train = PTG_Dataset(
                train_videos, self.hparams.num_classes, actions_dict, gt_path,
                features_path, self.hparams.sample_rate, self.hparams.window_size
            )

            self.data_val = PTG_Dataset(
                val_videos, self.hparams.num_classes, actions_dict, gt_path,
                features_path, self.hparams.sample_rate, self.hparams.window_size
            )

            self.data_test = PTG_Dataset(
                test_videos, self.hparams.num_classes, actions_dict, gt_path,
                features_path, self.hparams.sample_rate, self.hparams.window_size
            )
 

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        train_sampler = torch.utils.data.WeightedRandomSampler(
            self.data_train.weights, 
            self.hparams.epoch_length, 
            replacement=True, 
            generator=None
        )
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            sampler=train_sampler,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = MNISTDataModule()
