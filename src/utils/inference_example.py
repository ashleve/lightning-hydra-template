from src.models.mnist_model import LitModelMNIST
from src.transforms import mnist_transforms
from PIL import Image


def predict():
    """
    This is example of inference with a trained model.
    It Loads trained image classification model from checkpoint.
    Then it loads example image and predicts its label.
    """

    # ckpt can be also a URL!
    CKPT_PATH = "epoch=0.ckpt"

    # load model from checkpoint
    trained_model = LitModelMNIST.load_from_checkpoint(checkpoint_path=CKPT_PATH)

    # switch to evaluation mode
    trained_model.eval()
    trained_model.freeze()

    # load data
    img = Image.open("data/example_img.png").convert("L")  # convert to black and white
    # img = Image.open("data/example_img.png").convert("RGB")  # convert to RGB

    # preprocess
    img = mnist_transforms.mnist_test_transforms(img)
    img = img.reshape((1, *img.size()))  # reshape to form batch of size 1

    # inference
    output = trained_model(img)
    print(output)


if __name__ == "__main__":
    predict()
