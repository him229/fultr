import torch
from torch import nn


class ScoreModel(nn.Module):
    def __init__(self, clamp=False, masked_feat_id=None):
        super(ScoreModel, self).__init__()
        self.clamp = clamp
        self.masked_feat_id = masked_feat_id

    def compute_score(self, x):
        raise NotImplementedError

    def forward(self, x):
        if self.masked_feat_id is not None:
            x = x.index_fill(-1, torch.tensor(self.masked_feat_id, dtype=torch.long, device=x.device), 0.0)
        score = self.compute_score(x)
        if self.clamp:
            score = torch.clamp(score, -10, 10)
        return score


class LinearModel(ScoreModel):
    """
    One layer simple linear model
    """

    def __init__(self, input_dim=2, **kwargs):
        super(LinearModel, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.w = nn.Linear(input_dim, 1, bias=True)

    def compute_score(self, x):
        h = self.w(x)
        return h


class MLP(ScoreModel):
    def __init__(self, input_dim: int, hidden_layer: int, dropout: float = 0.0, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.input_size = input_dim
        fcs = []
        last_size = self.input_size
        for _ in range(hidden_layer):
            size = last_size // 2
            linear = nn.Linear(last_size, size)
            linear.bias.data.fill_(0.0)
            fcs.append(linear)
            last_size = size
            fcs.append(nn.ReLU())
            if dropout > 0.0:
                fcs.append(nn.Dropout(dropout))
        self.fc = nn.Sequential(*fcs)
        linear = nn.Linear(last_size, 1)
        linear.bias.data.fill_(0.0)
        self.final_layer = linear

    def compute_score(self, x):
        out = self.fc(x)
        out = self.final_layer(out)
        return out


def convert_vars_to_gpu(varlist):
    return [var.cuda() for var in varlist]
