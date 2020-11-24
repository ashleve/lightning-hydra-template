from torchvision import transforms


img_augmentation_transformations = [
    transforms.RandomRotation((-10, 10)),
]

train_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomApply(img_augmentation_transformations, 0.5)
])

test_preprocess = transforms.Compose([
    transforms.ToTensor()
])
