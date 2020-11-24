from torch import nn


class SimpleMNISTClassifier(nn.Module):

    def __init__(self, config):
        super().__init__()

        # mnist images are (1, 28, 28) (channels, width, height)
        self.model = nn.Sequential(
            nn.Linear(config["input_size"], config["lin1_size"]),
            nn.BatchNorm1d(config["lin1_size"]),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(config["lin1_size"], config["lin2_size"]),
            nn.BatchNorm1d(config["lin2_size"]),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(config["lin2_size"], config["lin3_size"]),
            nn.BatchNorm1d(config["lin3_size"]),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(config["lin3_size"], config["output_size"]),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)
        return self.model(x)
