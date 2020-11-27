from models.transfer_learning_img_classifier.lightning_module import LitModel
from data_modules import transforms
from PIL import Image


def predict():
    """
        Loads model from checkpoint.
        Model used in lightning_module.py and declaration of that model in models.py
        should be the same as during training.
    """

    CKPT_PATH = "epoch=0.ckpt"

    # load model from checkpoint
    trained_model = LitModel.load_from_checkpoint(checkpoint_path=CKPT_PATH)

    # switch to evaluation mode
    trained_model.eval()
    trained_model.freeze()

    # load data
    img = Image.open("../../example_img.png").convert("RGB")

    # preprocess
    img = transforms.imagenet_test_transforms(img)
    img = img.reshape((1, *img.size()))  # reshape to form batch of size 1

    # inference
    output = trained_model(img)
    print(output)


if __name__ == "__main__":
    predict()
