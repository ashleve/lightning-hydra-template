from torch import nn

class SimpleTransformerEncoderPooling(nn.Module):
    def __init__(
        self,
        input_size: int = 8,
        d_model: int = 64, 
        nhead: int = 8,
        num_layers: int = 6,
        output_size = 1
    ):
        super(SimpleTransformerEncoderPooling, self).__init__()

        self.input_size = input_size
        self.d_model = d_model
        self.nhead = nhead

        self.fc_in = nn.Linear(input_size, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.fc_in(x)
        x = self.transformer_encoder(x)
        x = x[:, 0, :]
        x = self.fc_out(x)
        return x.squeeze()