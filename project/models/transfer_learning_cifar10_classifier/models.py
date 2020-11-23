from pl_bolts.models.self_supervised import CPCV2
from efficientnet_pytorch import EfficientNet
from torchvision import models
from torch import nn


class EfficientNetPretrained(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.model = EfficientNet.from_pretrained('efficientnet-b1')

        for param in self.model.parameters():
            param.requires_grad = False

        self.model._fc = nn.Sequential(
            nn.Linear(self.model._fc.in_features, config["lin1_size"]),
            nn.BatchNorm1d(config["lin1_size"]),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(config["lin1_size"], config["lin2_size"]),
            nn.BatchNorm1d(config["lin2_size"]),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(config["lin2_size"], config["output_size"]),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)


class ResnetPretrained(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.model = models.resnet18(pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, config["lin1_size"]),
            nn.BatchNorm1d(config["lin1_size"]),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(config["lin1_size"], config["lin2_size"]),
            nn.BatchNorm1d(config["lin2_size"]),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(config["lin2_size"], config["output_size"]),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)


class ResnextPretrained(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.model = models.resnext101_32x8d(pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, config["lin1_size"]),
            nn.BatchNorm1d(config["lin1_size"]),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(config["lin1_size"], config["lin2_size"]),
            nn.BatchNorm1d(config["lin2_size"]),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(config["lin2_size"], config["output_size"]),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)


# this model doesn't work yet, we wait for lightning-bolts patch
# class ResnetPretrainedUnsupervised(nn.Module):
#     """
#         Trained without labels on Imagenet.
#         Perhaps the features when trained without labels are much better for classification or other tasks.
#     """
#     def __init__(self, config):
#         super().__init__()
#
#         self.model = CPCV2(encoder='resnet18', pretrained='imagenet128').encoder
#         # self.model = CPCV2(encoder='resnet18', pretrained='stl10').encoder
#
#         for param in self.model.parameters():
#             param.requires_grad = False
#
#         self.model.fc = nn.Sequential(
#             nn.Linear(self.model.fc.in_features, config["lin1_size"]),
#             nn.BatchNorm1d(config["lin1_size"]),
#             nn.ReLU(),
#             nn.Dropout(p=0.25),
#             nn.Linear(config["lin1_size"], config["lin2_size"]),
#             nn.BatchNorm1d(config["lin2_size"]),
#             nn.ReLU(),
#             nn.Dropout(p=0.2),
#             nn.Linear(config["lin2_size"], config["output_size"]),
#             nn.LogSoftmax(dim=1)
#         )
#
#     def forward(self, x):
#         return self.model(x)
