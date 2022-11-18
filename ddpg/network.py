import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class PreyNet(nn.Module):
    def __init__(self, n_inputs, n_outputs, learning_rate):
        super(PreyNet, self).__init__()
        self.out = nn.Linear(n_inputs, n_outputs)
        # training
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)

    def forward(self, x):
        # x = x.to(device)
        return self.out(x)

    def loss(self, q_outputs, q_targets):
        raise NotImplementedError


class PredatorNet(nn.Module):
    def __init__(self, n_inputs, n_outputs, learning_rate):
        super(PredatorNet, self).__init__()
        self.out = nn.Linear(n_inputs, n_outputs)
        # training
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)

    def forward(self, x):
        # x = x.to(device)
        return self.out(x)

    def loss(self, q_outputs, q_targets):
        raise NotImplementedError
