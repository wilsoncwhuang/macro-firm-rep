from torch import nn
import torch
from layers import FIM


class model(nn.Module):
    def __init__(self, feature_size=16, window_size=1, horizon_size=60, dropout=0, T_max=12, layer=False, batchNorm=True):
        super().__init__()
        self.layer = layer
        self.batchNorm = batchNorm
        self.bn = nn.BatchNorm1d(window_size * feature_size)

        self.hidden_layer_series = nn.Linear(
            window_size * feature_size, window_size * feature_size
        )   
        nn.init.xavier_uniform_(self.hidden_layer_series.weight)

        if self.batchNorm:
            self.net = nn.Sequential(
                self.hidden_layer_series,
                nn.BatchNorm1d(window_size * feature_size),
                FIM(window_size * feature_size, horizon=horizon_size),
            )
        else:
            self.net = nn.Sequential(
                self.hidden_layer_series,
                FIM(window_size * feature_size, horizon=horizon_size),
            )
      
        self.x_pos = nn.Embedding(T_max, feature_size)

    def forward(self, x):

        B, T, D = x.shape

        if self.layer == False:
            x_ids = torch.arange(T, device=x.device)
            x_pos_enc = self.x_pos(x_ids)
            x = x + x_pos_enc.unsqueeze(0) 

        x = x.view(x.shape[0], -1)
        if self.batchNorm:
            x = self.bn(x)
        return self.net(x)
