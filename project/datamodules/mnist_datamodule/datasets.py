from torch.utils.data import Dataset
from PIL import Image
import os


class TestDataset(Dataset):
    """
        Example dataset class for loading images from folder.
        'Dataset' classes can be used by 'DataModule' class.
    """

    def __init__(self, img_dir, transform):
        self.transform = transform
        self.images = [os.path.join(img_dir, fname) for fname in os.listdir(img_dir)]

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.images)
