from PIL import Image
from torchvision import transforms

from src.pl_models.mnist_model import MNISTLitModel


def predict():
    """Example of inference with trained model.
    It loads trained image classification model from checkpoint.
    Then it loads example image and predicts its label.
    """

    # ckpt can be also a URL!
    CKPT_PATH = "last.ckpt"

    # load model from checkpoint
    # model __init__ parameters will be loaded from ckpt automatically
    # you can also pass some parameter explicitly to override it
    trained_model = MNISTLitModel.load_from_checkpoint(checkpoint_path=CKPT_PATH)

    # print model hyperparameters
    print(trained_model.hparams)

    # switch to evaluation mode
    trained_model.eval()
    trained_model.freeze()

    # load data
    img = Image.open("data/example_img.png").convert("L")  # convert to black and white
    # img = Image.open("data/example_img.png").convert("RGB")  # convert to RGB

    # preprocess
    mnist_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((28, 28)),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    img = mnist_transforms(img)
    img = img.reshape((1, *img.size()))  # reshape to form batch of size 1

    # inference
    output = trained_model(img)
    print(output)


if __name__ == "__main__":
    predict()
