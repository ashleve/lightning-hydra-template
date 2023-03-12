from torch import nn

class SimpleTransformerEncoderPooling(nn.Module):
    def __init__(
        self,
        input_size: int = 8,
        d_model: int = 64,
        nhead: int = 8,
        dim_feedforward: int = 516,
        dropout: float = 0.1,
        num_layers: int = 6,
        output_size: int = 1
    ):
        super(SimpleTransformerEncoderPooling, self).__init__()

        self.input_size = input_size
        self.d_model = d_model
        self.nhead = nhead

        self.fc_in = nn.Linear(input_size, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model = d_model,
            nhead = nhead,
            dim_feedforward = dim_feedforward,
            dropout = dropout,
            batch_first=True
            )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, output_size)

    def forward(self, x, pad_mask = None):
        x = self.fc_in(x)
        x = self.transformer_encoder(x,src_key_padding_mask = pad_mask)
        x = x[:, 0, :]
        x = self.fc_out(x)
        return x.squeeze()