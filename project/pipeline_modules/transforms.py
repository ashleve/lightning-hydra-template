from torchvision import transforms


data_augmentation_transformations = [
    transforms.RandomAffine((-15, 15), translate=(0.2, 0.2)),
    transforms.RandomHorizontalFlip(p=0.8),
    transforms.RandomRotation((-15, 15)),
]

train_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomApply(data_augmentation_transformations, 0.8),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
