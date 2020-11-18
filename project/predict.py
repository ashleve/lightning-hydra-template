from pipeline_modules.lightning_wrapper import LitModel
from pipeline_modules import transforms
import pytorch_lightning as pl
from train import load_config
import torch


def predict():
    # load config
    config = load_config()

    # load model from checkpoint
    pretrained_model = LitModel.load_from_checkpoint("example.ckpt", config=config["hparams"])
    pretrained_model.eval()

    # load data
    img = None

    # preprocess
    img = transforms.efficient_net_test_preprocess(img)
    img = img.reshape((1, *img.size()))  # reshape to form batch of size 1

    # inference
    output = pretrained_model(img)
    print(output)


def download_from_wanb():
    MODEL_PATH = ""
    CODE_PATH = ""


if __name__ == "__main__":
    predict()
