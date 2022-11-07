from pettingzoo.mpe import simple_tag_v2
# from gymnasium.utils.save_video import save_video

import time
import random
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Tag:
    def __init__(self, render=False):
        self.env = simple_tag_v2.env(
            num_good=1,
            num_adversaries=3,
            num_obstacles=2,
            max_cycles=100,
            continuous_actions=False,
            render_mode='human' if render else ''
        )

        self.env.reset()
        for agent in self.env.agent_iter():
            observation, reward, termination, truncation, info = self.env.last()
            action = None if termination or truncation else self.env.action_space(
                agent).sample()  # this is where you would insert your policy

            self.env.step(action)
            time.sleep(0.05)

        self.env.render()
        self.env.close()

    def train(self):
        # train Deep Q-network

        num_episodes = 1000
        episode_limit = 100
        batch_size = 64
        learning_rate = 0.005
        gamma = 0.99  # discount rate
        tau = 0.01  # target network update rate
        replay_memory_capacity = 10000
        prefill_memory = True
        val_freq = 100  # validation frequency

        n_inputs = self.env.observation_space.n
        n_outputs = self.env.action_space.n

        # initialize DQN and replay memory
        policy_dqn = DQN(n_inputs, n_outputs, learning_rate)
        target_dqn = DQN(n_inputs, n_outputs, learning_rate)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        replay_memory = ReplayMemory(replay_memory_capacity)

        # prefill replay memory with random actions
        if prefill_memory:
            print('prefill replay memory')
            s = self.env.reset()
            while replay_memory.count() < replay_memory_capacity:
                a = self.env.action_space.sample()
                s1, r, d, _ = self.env.step(a)
                replay_memory.add(s, a, r, s1, d)
                s = s1 if not d else self.env.reset()

        # training loop
        try:
            print('start training')
            epsilon = 1.0
            rewards, lengths, losses, epsilons = [], [], [], []
            for i in range(num_episodes):
                # init new episode
                s, ep_reward, ep_loss = self.env.reset(), 0, 0
                for j in range(episode_limit):
                    # select action with epsilon-greedy strategy
                    if np.random.rand() < epsilon:
                        a = self.env.action_space.sample()
                    else:
                        with torch.no_grad():
                            a = policy_dqn(torch.from_numpy(self.one_hot([s], n_inputs)).float()).argmax().item()
                    # perform action
                    s1, r, d, _ = self.env.step(a)
                    # store experience in replay memory
                    replay_memory.add(s, a, r, s1, d)
                    # batch update
                    if replay_memory.count() >= batch_size:
                        # sample batch from replay memory
                        batch = np.array(replay_memory.sample(batch_size), dtype=int)
                        ss, aa, rr, ss1, dd = batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3], batch[:, 4]
                        # do forward pass of batch
                        policy_dqn.optimizer.zero_grad()
                        Q = policy_dqn(torch.from_numpy(self.one_hot(ss, n_inputs)).float())
                        # use target network to compute target Q-values
                        with torch.no_grad():
                            # TODO: use target net
                            Q1 = target_dqn(torch.from_numpy(self.one_hot(ss1, n_inputs)).float())
                        # compute target for each sampled experience
                        q_targets = Q.clone()
                        for k in range(batch_size):
                            q_targets[k, aa[k]] = rr[k] + gamma * Q1[k].max().item() * (not dd[k])
                        # update network weights
                        loss = policy_dqn.loss(Q, q_targets)
                        loss.backward()
                        policy_dqn.optimizer.step()
                        # update target network parameters from policy network parameters
                        target_dqn.update_params(policy_dqn.state_dict(), tau)
                    else:
                        loss = 0
                    # bookkeeping
                    s = s1
                    ep_reward += r
                    ep_loss += loss.item()
                    if d:
                        break
                # bookkeeping
                epsilon *= num_episodes / (i / (num_episodes / 20) + num_episodes)  # decrease epsilon
                epsilons.append(epsilon)
                rewards.append(ep_reward)
                lengths.append(j + 1)
                losses.append(ep_loss)
                if (i + 1) % val_freq == 0:
                    print('%5d mean training reward: %5.2f' % (i + 1, np.mean(rewards[-val_freq:])))
            print('done')
        except KeyboardInterrupt:
            print('interrupt')


# one-hot encoder for the states
def one_hot(self, i, l):
    a = np.zeros((len(i), l))
    a[range(len(i)), i] = 1
    return a


class DQN(nn.Module):
    """Deep Q-network with target network"""

    def __init__(self, n_inputs, n_outputs, learning_rate):
        super(DQN, self).__init__()
        # network
        self.out = nn.Linear(n_inputs, n_outputs)
        # training
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)

    def forward(self, x):
        return self.out(x)

    def loss(self, q_outputs, q_targets):
        return torch.sum(torch.pow(q_targets - q_outputs, 2))

    def update_params(self, new_params, tau):
        params = self.state_dict()
        for k in params.keys():
            params[k] = (1 - tau) * params[k] + tau * new_params[k]
        self.load_state_dict(params)


class ReplayMemory(object):
    """Experience Replay Memory"""

    def __init__(self, capacity):
        # self.size = size
        self.memory = deque(maxlen=capacity)

    def add(self, *args):
        """Add experience to memory."""
        self.memory.append([*args])

    def sample(self, batch_size):
        """Sample batch of experiences from memory with replacement."""
        return random.sample(self.memory, batch_size)

    def count(self):
        return len(self.memory)


if __name__ == "__main__":
    test = Tag(render=True)
