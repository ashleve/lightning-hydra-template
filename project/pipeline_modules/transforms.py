from torchvision import transforms


data_augmentation_transformations = [
    transforms.RandomAffine((-15, 15), translate=(0.2, 0.2)),
    transforms.RandomHorizontalFlip(p=0.65),
    transforms.RandomRotation((-15, 15)),
    transforms.ColorJitter(hue=.05, saturation=.05),
]

efficient_net_train_preprocess = transforms.Compose([
    transforms.Resize((256, 256)),  # the correct size for eff net b1 is 224x224 but we random crop it later
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.RandomApply(data_augmentation_transformations, 0.8),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

efficient_net_test_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
