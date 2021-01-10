"""
Example file containing data transformations, which can be used by datamodule.
"""
from torchvision import transforms


mnist_train_transforms = transforms.Compose([
    transforms.ToTensor(),
])

mnist_test_transforms = transforms.Compose([
    transforms.ToTensor(),
])
