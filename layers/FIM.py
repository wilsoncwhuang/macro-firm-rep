import torch
from torch import nn


class PoissonProb(nn.Module):
    def __init__(self, period=12):
        super(PoissonProb, self).__init__()
        self.period = period

    def forward(self, input):
        return 1 - torch.exp(-input / self.period)


# class Exp(nn.Module):
#     def forward(self, x):
#         return torch.exp(x)


class FIM(nn.Module):
    def __init__(self, input_dim, horizon=60, CRI=True):
        super(FIM, self).__init__()
        self.horizon = horizon
        self.CRI = CRI

        def_linear = nn.Linear(input_dim, horizon)
        nn.init.xavier_normal_(def_linear.weight)  # glorot_normal init
        oth_linear = nn.Linear(input_dim, horizon)
        nn.init.xavier_normal_(oth_linear.weight)  # glorot_normal init

        self.def_net = nn.Sequential(def_linear, nn.Softplus(), PoissonProb())

        self.oth_net = nn.Sequential(oth_linear, nn.Softplus(), PoissonProb())

    def forward(self, input):
        x_def = self.def_net(input)
        x_oth = self.oth_net(input)
        if self.CRI:
            x_non_def = 1 - x_def
            x_oth = x_oth * x_non_def
        x_sur = 1 - (x_def + x_oth)
        result = torch.stack((x_sur, x_def, x_oth), dim=-1)
        return result
