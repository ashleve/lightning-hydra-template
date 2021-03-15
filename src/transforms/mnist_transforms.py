from torchvision import transforms

mnist_train_transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

mnist_test_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((28, 28)),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)
