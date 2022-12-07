import torch
import random
import copy
from torch.functional import F
from collections import deque
import numpy as np


def extract_data(sample, device):
    states_tuple = np.array([_[0] for _ in sample])
    actions_tuple = np.array([_[1] for _ in sample])
    next_states_tuple = np.array([_[2] for _ in sample])
    rewards_tuple = np.float32(np.array([_[3] for _ in sample]))

    compressed_states = torch.from_numpy(states_tuple).requires_grad_()
    compressed_actions = torch.from_numpy(actions_tuple).requires_grad_()
    compressed_next_states = torch.from_numpy(next_states_tuple)
    compressed_rewards = torch.from_numpy(rewards_tuple)

    return F.normalize(compressed_states).to(device), F.normalize(compressed_actions).to(device), \
           F.normalize(compressed_next_states).to(device), compressed_rewards.to(device)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.state = None
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.randn() for i in range(len(x))])
        self.state = x + dx
        return self.state


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
