from torch import nn


class SimpleDenseNet(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(hparams["input_size"], hparams["lin1_size"]),
            nn.BatchNorm1d(hparams["lin1_size"]),
            nn.ReLU(),
            nn.Dropout(p=hparams["dropout1"]),
            nn.Linear(hparams["lin1_size"], hparams["lin2_size"]),
            nn.BatchNorm1d(hparams["lin2_size"]),
            nn.ReLU(),
            nn.Dropout(p=hparams["dropout2"]),
            nn.Linear(hparams["lin2_size"], hparams["lin3_size"]),
            nn.BatchNorm1d(hparams["lin3_size"]),
            nn.ReLU(),
            nn.Dropout(p=hparams["dropout3"]),
            nn.Linear(hparams["lin3_size"], hparams["output_size"]),
        )

    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # mnist images are (1, 28, 28) (channels, width, height)
        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)

        return self.model(x)
