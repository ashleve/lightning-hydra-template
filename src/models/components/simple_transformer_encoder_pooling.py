from torch import nn
import torch

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
        self.fc_out = nn.Linear(d_model, output_size) # 3 for min, max, mean pooling

    def forward(self, x, pad_mask = None):
        x = self.fc_in(x)
        x = self.transformer_encoder(x,src_key_padding_mask = pad_mask)
        # x = x[:, 0, :]
        # Min, max, and mean pooling
        # x_min, _ = torch.min(x, dim=1)
        # x_max, _ = torch.max(x, dim=1)
        # x_mean = torch.mean(x, dim=1)

        # Concatenate the pooled features
        # x_pooled = torch.cat((x_min, x_max, x_mean), dim=1)

        x = self.fc_out(x) #x_pooled
        return x.squeeze()
    

if __name__ == "__main__":
    _ = SimpleTransformerEncoderPooling()