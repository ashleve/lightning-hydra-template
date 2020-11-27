from torch import nn


class SimpleMNISTClassifier(nn.Module):

    def __init__(self, hparams):
        super().__init__()

        # mnist images are (1, 28, 28) (channels, width, height)
        self.model = nn.Sequential(
            nn.Linear(hparams["input_size"], hparams["lin1_size"]),
            nn.BatchNorm1d(hparams["lin1_size"]),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hparams["lin1_size"], hparams["lin2_size"]),
            nn.BatchNorm1d(hparams["lin2_size"]),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(hparams["lin2_size"], hparams["lin3_size"]),
            nn.BatchNorm1d(hparams["lin3_size"]),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hparams["lin3_size"], hparams["output_size"]),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)
        return self.model(x)
