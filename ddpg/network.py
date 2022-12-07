import torch.nn as nn
import torch


class ActorNet(nn.Module):
    def __init__(self, d_in, d_out, batch_size):
        super(ActorNet, self).__init__()
        self.action_vector = nn.Sequential(nn.Linear(d_in, batch_size),
                                           nn.ReLU(),
                                           nn.Linear(batch_size, batch_size // 2),
                                           nn.ReLU(),
                                           nn.Linear(batch_size // 2, d_out),
                                           nn.Sigmoid())

    def forward(self, x):
        return self.action_vector(x)


class CriticNet(nn.Module):
    def __init__(self, d_in, d_out, batch_size):
        super(CriticNet, self).__init__()
        self.value = nn.Sequential(nn.Linear(d_in + d_out, batch_size),
                                   nn.ReLU(),
                                   nn.Linear(batch_size, batch_size // 2),
                                   nn.ReLU(),
                                   nn.Linear(batch_size // 2, 1))

    def forward(self, x):
        return self.value(x)


class Actor:
    def __init__(self, n_o, n_a, batch_size):
        self.policy = ActorNet(n_o, n_a, batch_size)

    def get_action(self, state):
        return self.policy(state)


class Critic:
    def __init__(self, n_o, n_a, batch_size):
        self.value_func = CriticNet(n_o, n_a, batch_size)

    def get_state_value(self, state, action):
        x = torch.cat((state, action), dim=1)
        return self.value_func.forward(x)


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)
