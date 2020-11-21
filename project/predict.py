from training_modules.lightning_wrapper import LitModel
from training_modules.data_modules import *
from training_modules.datasets import *
from training_modules import transforms
from PIL import Image


def predict():
    """
        Loads model from checkpoint.
        Files training_modules/models.py and training_modules/lightning_wrapper.py should be the same.
    """

    CKPT_PATH = "epoch=0.ckpt"

    # load model from checkpoint
    pretrained_model = LitModel.load_from_checkpoint(checkpoint_path=CKPT_PATH)

    # switch to evaluation mode
    pretrained_model.eval()
    pretrained_model.freeze()

    # load data
    img = Image.open("example_img.png").convert("RGB")

    # preprocess
    img = transforms.efficient_net_test_preprocess(img)
    img = img.reshape((1, *img.size()))  # reshape to form batch of size 1

    # inference
    output = pretrained_model(img)
    print(output)


if __name__ == "__main__":
    predict()
