import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Transformer1d(nn.Module):
    """

    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)

    Output:
        out: (n_samples, n_length)

    Pararmetes:

    """

    def __init__(self, d_model, nhead, num_layers=4):
        super(Transformer1d, self).__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

    def forward(self, x):

        out = self.transformer_encoder(x)

        return out

if __name__ == '__main__':
    # x = torch.randn(10, 1, 10)
    encoder_layer = nn.TransformerEncoderLayer(d_model=120, nhead=4)
    encoder_layer = nn.TransformerEncoder(encoder_layer, num_layers=6)
    src = torch.rand(10, 1, 120)
    out = encoder_layer(src)
    out = out.squeeze(1)

    # model = Transformer1d(2, 120, 8)
    # out = model(x)
    print(out.shape)
