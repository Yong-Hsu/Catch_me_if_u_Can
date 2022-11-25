import torch
import random
import math
from torch.functional import F
from collections import deque
import numpy as np


def extract_data(sample):
    states_tuple = np.array([_[0] for _ in sample])
    actions_tuple = np.array([_[1] for _ in sample])
    next_states_tuple = np.array([_[2] for _ in sample])
    rewards_tuple = np.float32(np.array([_[3] for _ in sample]))

    compressed_states = torch.from_numpy(states_tuple).requires_grad_()
    compressed_actions = torch.from_numpy(actions_tuple).requires_grad_()
    compressed_next_states = torch.from_numpy(next_states_tuple)
    compressed_rewards = torch.from_numpy(rewards_tuple)

    return F.normalize(compressed_states), F.normalize(compressed_actions), F.normalize(compressed_next_states), \
        compressed_rewards


class OuNoise:
    def __init__(self, th=1, mu=0, sig=1, dt=1):
        self.t = None
        self.theta = th
        self.mu = mu
        self.sigma = sig
        self.dt = dt
        self.x = 0

    def reset(self):
        self.t = 0

    def step(self):
        x_1 = self.x + self.theta * (self.mu - self.x) * self.dt + self.sigma * math.sqrt(
            self.dt) * random.normalvariate(0, 1)
        self.x = x_1
        return self.x


class ReplayBuffer:
    def __init__(self, max_length):
        self.max_length = max_length
        self.Buffer = deque(
            maxlen=self.max_length)  # The last first entry gets automatically removed when the buffer size is exceeded

    def append_memory(self, experience):
        self.Buffer.appendleft(experience)

    def sample(self, batch_size=32):
        return random.sample(self.Buffer, batch_size)

    def buf_len(self):
        return len(self.Buffer)
