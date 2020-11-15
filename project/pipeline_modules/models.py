from torch import nn
import torch.nn.functional as F


class ModelMNISTv1(nn.Module):

    def __init__(self, config):
        super().__init__()

        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = nn.Linear(28 * 28, config["lin1_size"])
        self.layer_2 = nn.Linear(config["lin1_size"], config["lin2_size"])
        self.layer_3 = nn.Linear(config["lin2_size"], 10)

    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)

        return F.log_softmax(x, dim=1)


class ModelMNISTv2(nn.Module):

    def __init__(self, config):
        super().__init__()

        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = nn.Linear(28 * 28, config["lin1_size"])
        self.layer_2 = nn.Linear(config["lin1_size"], config["lin2_size"])
        self.layer_3 = nn.Linear(config["lin2_size"], config["lin3_size"])
        self.layer_4 = nn.Linear(config["lin3_size"], 10)

    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)
        x = F.relu(x)
        x = self.layer_4(x)

        return F.log_softmax(x, dim=1)
