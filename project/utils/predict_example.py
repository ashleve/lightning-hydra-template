from data_modules.mnist_digits_datamodule import transforms
from PIL import Image

# the LitModel you import should be the same as the one you used for training!
from models.simple_mnist_classifier.lightning_module import LitModel


def predict():
    """
        This method is example of inference with a trained model.
        It Loads trained image classification model from checkpoint.
        Then it loads example image and predicts its label.
        Model used in lightning_module.py should be the same as during training!!!
    """

    CKPT_PATH = "epoch=0.ckpt"

    # load model from checkpoint
    trained_model = LitModel.load_from_checkpoint(checkpoint_path=CKPT_PATH)

    # switch to evaluation mode
    trained_model.eval()
    trained_model.freeze()

    # load data
    img = Image.open("data/example_img.png").convert("L")  # for monochromatic conversion
    # img = Image.open("data/example_img.png").convert("RGB")  # for RGB conversion

    # preprocess
    img = transforms.mnist_test_transforms(img)
    img = img.reshape((1, *img.size()))  # reshape to form batch of size 1

    # inference
    output = trained_model(img)
    print(output)


if __name__ == "__main__":
    predict()
