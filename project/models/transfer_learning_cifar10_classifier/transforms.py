from torchvision import transforms


img_augmentation_transformations = [
    transforms.RandomAffine((-15, 15), translate=(0.2, 0.2)),
    transforms.RandomHorizontalFlip(p=0.65),
    transforms.RandomRotation((-15, 15)),
    transforms.ColorJitter(hue=.05, saturation=.05),
]

train_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomApply(img_augmentation_transformations, 0.5)
])

test_preprocess = transforms.Compose([
    transforms.ToTensor()
])
