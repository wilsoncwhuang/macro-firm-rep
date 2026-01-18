import torch
from torch import nn
from layers import FIM
from torch.nn import functional as F

class model(nn.Module):
    def __init__(
        self, feature_size=16, window_size=1, horizon_size=60, dropout=0, T_max=12, layer=False
    ):
        super().__init__()
        self.layer = layer

        hidden_layer1 = nn.Linear(
            window_size * feature_size, window_size * feature_size
        )
        nn.init.xavier_uniform_(hidden_layer1.weight)
        hidden_layer2 = nn.Linear(
            window_size * feature_size, window_size * feature_size
        )
        nn.init.xavier_uniform_(hidden_layer2.weight)

        self.net = nn.Sequential(
            hidden_layer1,
            nn.BatchNorm1d(window_size * feature_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            hidden_layer2,
            nn.BatchNorm1d(window_size * feature_size),
            nn.ReLU(),
            nn.Dropout(dropout),
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
        result = self.net(x)
        return result
