from pipeline_modules.lightning_wrapper import LitModel
from pipeline_modules import transforms
import pytorch_lightning as pl
from train import load_config
import torch


# load config
config = load_config()

# load model from checkpoint
pretrained_model = LitModel.load_from_checkpoint("example.ckpt", config=config)
pretrained_model.eval()

# load data
img = None

# preprocess
img = transforms.efficient_net_test_preprocess(img)
img = img.reshape((1, *img.size()))  # reshape to form batch of size 1

# inference
output = pretrained_model(img)
print(output)
