from torch import nn


class ExampleModel(nn.Module):

    def __init__(self, hparams):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(hparams["input_size"], hparams["lin1_size"]),
            nn.BatchNorm1d(hparams["lin1_size"]),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(hparams["lin1_size"], hparams["lin2_size"]),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)
        return self.model(x)
