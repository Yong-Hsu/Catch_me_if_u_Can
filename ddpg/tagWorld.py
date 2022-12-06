import time

from pettingzoo.mpe import simple_tag_v2

import copy
import random
from utils import ReplayBuffer, extract_data
from tqdm import tqdm
import matplotlib.pyplot as plt

# import torch
import numpy as np
from network import *


class TagWorld:
    def __init__(self):
        self.n_good = 1
        self.n_adv = 3
        self.n_obstacles = 0
        self.max_cycs = 1000
        self.continuous = True

        self.env = simple_tag_v2.env(
            # todo: works with one good agent now
            num_good=self.n_good,
            num_adversaries=self.n_adv,
            num_obstacles=self.n_obstacles,
            max_cycles=self.max_cycs,
            continuous_actions=self.continuous
        )

        # parameters
        self.GAMMA = 0.99  # Discount factor
        self.ALPHA = 1e-3  # learning rate
        self.TAU = 0.001
        self.BATCH_SIZE = 32
        self.BUFFER_SIZE = 5000
        self.epsilon = 0.7
        self.decay = 0.99999
        self.max_episodes = 26
        self.max_rollout = 500

        n_inputs_good = 10
        n_inputs_adv = 12
        n_outputs = 5

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Running on device:", self.device)

        # initiate the networks for the agents
        # the reason why critic network use 1 as output and the unsqueeze later
        self.GoodNetActor = Actor(n_inputs_good, n_outputs, self.BATCH_SIZE)
        self.GoodNetCritic = Critic(n_inputs_good, 1, self.BATCH_SIZE)

        self.AdvNetActor = Actor(n_inputs_adv, n_outputs, self.BATCH_SIZE)
        self.AdvNetCritic = Critic(n_inputs_adv, 1, self.BATCH_SIZE)

        # Target Network
        self.GoodNetActorTarget = copy.deepcopy(self.GoodNetActor)
        self.GoodNetCriticTarget = copy.deepcopy(self.GoodNetCritic)

        self.AdvNetActorTarget = copy.deepcopy(self.AdvNetActor)
        self.AdvNetCriticTarget = copy.deepcopy(self.AdvNetCritic)

        # to GPU
        self.GoodNetActor.policy.to(self.device)
        self.GoodNetCritic.value_func.to(self.device)
        self.AdvNetActor.policy.to(self.device)
        self.AdvNetCritic.value_func.to(self.device)
        self.GoodNetActorTarget.policy.to(self.device)
        self.GoodNetCriticTarget.value_func.to(self.device)
        self.AdvNetActorTarget.policy.to(self.device)
        self.AdvNetCriticTarget.value_func.to(self.device)

        # Experience
        self.ReplayBufferGood = ReplayBuffer(self.BUFFER_SIZE)
        self.ReplayBufferAdv = ReplayBuffer(self.BUFFER_SIZE)

        # optimizer todo:
        self.optim_good = torch.optim.Adam(self.GoodNetCritic.value_func.parameters(), lr=self.ALPHA)
        self.optim_adv = torch.optim.Adam(self.AdvNetCritic.value_func.parameters(), lr=self.ALPHA)

        self.env.reset()

    def train(self):
        self.env.reset()

        # Store rewards to be plotted
        rewards_good = []
        rewards_adv = []

        # Epsilon stored for plotting
        epsilon_plotting = []

        # Loss for plotting
        loss_plotting_good = []
        loss_plotting_adv = []

        for _ in tqdm(range(self.max_episodes)):
            self.env.reset()
            count = 0

            reward_good = 0
            reward_adv = 0

            output_good = 0
            output_adv = 0
            # Decay greedy epsilon
            self.epsilon = self.epsilon * self.decay

            for agent in self.env.agent_iter():
                # if count % 50 == 0:
                #     print(count)

                # action from network
                observation, agent_reward, _, _, _ = self.env.last()
                if agent == 'agent_0':
                    action = self.GoodNetActor.get_action(torch.from_numpy(observation).to(self.device))
                    action = action.cpu().detach().numpy()
                    action = np.clip(action, 0, 1)  # clip negative and bigger than 1 values

                    agent_reward += 2 * (np.linalg.norm((observation[4], observation[5])) +
                                         np.linalg.norm((observation[6], observation[7])) +
                                         np.linalg.norm((observation[8], observation[9])))
                    agent_reward -= min(abs(1 // (5 * np.linalg.norm((observation[0], observation[1])))), 50)
                    reward_good = reward_good + agent_reward
                else:
                    action = self.AdvNetActor.get_action(torch.from_numpy(observation).to(self.device))
                    action = action.cpu().detach().numpy()
                    action = np.clip(action, 0, 1)

                    agent_reward -= 6 * np.linalg.norm((observation[8], observation[9]))
                    agent_reward -= min(abs(1 // (5 * np.linalg.norm((observation[0], observation[1])))), 50)
                    reward_adv = reward_adv + agent_reward

                # epsilon greedy, if true, replace the action above
                p = random.random()
                if p < self.epsilon:
                    action = self.env.action_space(agent).sample()
                # Decay greedy epsilon
                # self.epsilon = self.epsilon * self.decay

                # Get the new state, reward, and done signal
                self.env.step(action)
                _, _, termination_new, truncation_new, _ = self.env.last()
                observation_new = self.env.observe(agent)
                reward_new = self.env.rewards[agent]
                if agent == 'agent_0':
                    reward_new += (np.linalg.norm((observation_new[4], observation_new[5])) +
                                   np.linalg.norm((observation_new[6], observation_new[7])) +
                                   np.linalg.norm((observation_new[8], observation_new[9])))
                    reward_new -= min(abs(1 // (5 * np.linalg.norm((observation_new[0], observation_new[1])))), 50)
                else:
                    reward_new -= 3 * np.linalg.norm((observation_new[8], observation_new[9]))
                    reward_new -= min(abs(1 // (5 * np.linalg.norm((observation_new[0], observation_new[1])))), 50)

                # store replay buffer
                experience = [observation, action, observation_new, reward_new]
                if agent == 'agent_0':
                    # different iteration or do the calculation
                    self.ReplayBufferGood.append_memory(experience)
                else:
                    self.ReplayBufferAdv.append_memory(experience)

                if termination_new or truncation_new or count > self.max_rollout:
                    break

                # run all agents in every loop
                if self.ReplayBufferAdv.buf_len() >= self.BUFFER_SIZE:
                    # sampled_experience_good = self.ReplayBufferGood.sample()
                    sampled_experience_adv = self.ReplayBufferAdv.sample(batch_size=64)

                    # Adversarial Agent
                    compressed_states_adv, compressed_actions_adv, compressed_next_states_adv, compressed_rewards_adv \
                        = extract_data(sampled_experience_adv, self.device)

                    target_action_adv = self.AdvNetActorTarget.get_action(compressed_next_states_adv)
                    target_action_adv = target_action_adv.mean(dim=1).unsqueeze(-1)
                    target_value_adv = self.AdvNetCriticTarget.get_state_value(compressed_next_states_adv,
                                                                               target_action_adv)

                    # compute targets
                    # target_v_good = compressed_rewards_good.unsqueeze(1) + self.GAMMA * target_value_good
                    target_v_adv = compressed_rewards_adv.unsqueeze(1) + self.GAMMA * target_value_adv

                    # calculate Q function
                    compressed_actions_adv = compressed_actions_adv.mean(dim=1).unsqueeze(-1)
                    actual_v_adv = self.AdvNetCritic.get_state_value(compressed_states_adv, compressed_actions_adv)

                    # train the network
                    loss = nn.MSELoss()
                    # output_good = loss(actual_v_good, target_v_good)
                    output_adv = loss(actual_v_adv, target_v_adv)

                    # self.optim_good.zero_grad()
                    self.optim_adv.zero_grad()
                    # output_good.backward(retain_graph=True)
                    output_adv.backward(retain_graph=True)
                    # self.optim_good.step()
                    self.optim_adv.step()

                    # self.GoodNetCritic.value_func.zero_grad()
                    self.AdvNetCritic.value_func.zero_grad()

                    for s, a in zip(compressed_states_adv.split(1), compressed_actions_adv.split(1)):
                        online_v_adv = self.AdvNetCritic.get_state_value(s, a)
                        grad_wrt_a_adv = torch.autograd.grad(online_v_adv, (s, a))

                        action_adv = self.AdvNetActor.get_action(s)
                        action_adv.mean().backward(retain_graph=True)

                        for param in self.AdvNetActor.policy.parameters():
                            param.data += self.ALPHA * (param.grad * grad_wrt_a_adv[1].item()) / self.BATCH_SIZE

                        self.AdvNetActor.policy.zero_grad()
                        self.AdvNetCritic.value_func.zero_grad()

                    # Adversarial agent
                    for param_o_adv, param_t_adv in zip(self.AdvNetActor.policy.parameters(),
                                                        self.AdvNetActorTarget.policy.parameters()):
                        param_t_adv.data = param_o_adv.data * self.TAU + param_t_adv.data * (1 - self.TAU)

                    for param_o_adv, param_t_adv in zip(self.AdvNetCritic.value_func.parameters(),
                                                        self.AdvNetCriticTarget.value_func.parameters()):
                        param_t_adv.data = param_o_adv.data * self.TAU + param_t_adv.data * (1 - self.TAU)

                    self.AdvNetActor.policy.zero_grad()
                    self.AdvNetActorTarget.policy.zero_grad()
                    self.AdvNetCritic.value_func.zero_grad()
                    self.AdvNetCriticTarget.value_func.zero_grad()

                # if self.ReplayBufferGood.buf_len() >= self.BUFFER_SIZE:
                #     #
                #     sampled_experience_good = self.ReplayBufferGood.sample()
                #
                #     # calculate target network
                #     # Good Agent
                #     compressed_states_good, compressed_actions_good, compressed_next_states_good, \
                #     compressed_rewards_good = extract_data(sampled_experience_good, self.device)
                #
                #     target_action_good = self.GoodNetActorTarget.get_action(compressed_next_states_good)
                #     # todo: should we average the action tensor? Maybe we should use all those info, same below
                #     target_action_good = target_action_good.mean(dim=1).unsqueeze(-1)  # (32, 5) -> (32, 1)
                #     target_value_good = self.GoodNetCriticTarget.get_state_value(compressed_next_states_good,
                #                                                                  target_action_good)
                #
                #     # Compute targets
                #     target_v_good = compressed_rewards_good.unsqueeze(1) + self.GAMMA * target_value_good
                #
                #     # calculate Q function
                #     compressed_actions_good = compressed_actions_good.mean(dim=1).unsqueeze(-1)
                #     actual_v_good = self.GoodNetCritic.get_state_value(compressed_states_good, compressed_actions_good)
                #
                #     # train the network
                #     loss = nn.MSELoss()
                #     output_good = loss(actual_v_good, target_v_good)
                #
                #     self.optim_good.zero_grad()
                #     output_good.backward(retain_graph=True)
                #     self.optim_good.step()
                #     self.GoodNetCritic.value_func.zero_grad()
                #
                #     for s, a in zip(compressed_states_good.split(1), compressed_actions_good.split(1)):
                #         online_v_good = self.GoodNetCritic.get_state_value(s, a)
                #         grad_wrt_a_good = torch.autograd.grad(online_v_good, (s, a))
                #
                #         action_good = self.GoodNetActor.get_action(s)
                #         action_good.mean().backward(retain_graph=True)
                #
                #         for param in self.GoodNetActor.policy.parameters():
                #             param.data += self.ALPHA * (param.grad * grad_wrt_a_good[1].item()) / self.BATCH_SIZE
                #
                #         self.GoodNetActor.policy.zero_grad()
                #         self.GoodNetCritic.value_func.zero_grad()
                #
                #     # Good agent
                #     for param_o_good, param_t_good in zip(self.GoodNetActor.policy.parameters(),
                #                                           self.GoodNetActorTarget.policy.parameters()):
                #         param_t_good.data = param_o_good.data * self.TAU + param_t_good.data * (1 - self.TAU)
                #
                #     for param_o_good, param_t_good in zip(self.GoodNetCritic.value_func.parameters(),
                #                                           self.GoodNetCriticTarget.value_func.parameters()):
                #         param_t_good.data = param_o_good.data * self.TAU + param_t_good.data * (1 - self.TAU)
                #
                #     self.GoodNetActor.policy.zero_grad()
                #     self.GoodNetActorTarget.policy.zero_grad()
                #     self.GoodNetCritic.value_func.zero_grad()
                #     self.GoodNetCriticTarget.value_func.zero_grad()
                #
                # torch.save(self.GoodNetActorTarget.policy.state_dict(), 'good_target_actor_state_1.pt')
                # torch.save(self.GoodNetCriticTarget.value_func.state_dict(), 'good_target_critic_state_1.pt')

                count = count + 1

            rewards_good.append(reward_good / self.n_good)  # appends the average reward of all good agents
            rewards_adv.append(reward_adv / self.n_adv)  # appends the average reward of all the adversarial agents

            epsilon_plotting.append(self.epsilon)  # appends the epsilon value after each episode

            loss_plotting_good.append(output_good)
            loss_plotting_adv.append(output_adv)

        torch.save(self.AdvNetActor.policy.state_dict(), f'AdvNetActor_{time.time()}.pt')
        # torch.save(self.AdvNetActorTarget.policy.state_dict(), f'adv_target_actor_state_{time.time()}.pt')
        # torch.save(self.AdvNetCriticTarget.value_func.state_dict(), f'adv_target_critic_state_{time.time()}.pt')
        self.plot_res(rewards_good, rewards_adv, epsilon_plotting, loss_plotting_good, loss_plotting_adv)

    @staticmethod
    def plot_res(rewards_good, rewards_adv, epsilon_plotting, loss_plotting_good, loss_plotting_adv):
        # Plotting the avg rewards
        plt.figure(1)
        plt.plot(rewards_good)
        plt.title('Good agents average rewards')
        plt.xlabel('Number of episodes')
        plt.ylabel('Average Reward')
        plt.savefig(f'fig_{time.time()}_1.png')

        plt.figure(2)
        plt.plot(rewards_adv)
        plt.title('Adversial agents average rewards')
        plt.xlabel('Number of episodes')
        plt.ylabel('Average Reward')
        plt.savefig(f'fig_{time.time()}_2.png')

        plt.figure(3)
        plt.plot(epsilon_plotting)
        plt.title('Epsilon Decay')
        plt.xlabel('Number of episodes')
        plt.ylabel('Epsilon value')
        plt.savefig(f'fig_{time.time()}_3.png')

        plt.show()

        # fig1, axes1 = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
        # plt.figure(figsize=(16, 9))
        # fig1.subplots_adjust(hspace=.5)
        # axes1[0].set(title='Good agents average rewards', xlabel='Number of episodes', ylabel='Average Reward')
        # axes1[1].set(title='Adversial agents average rewards', xlabel='Number of episodes', ylabel='Average Reward')
        # axes1[0].plot(rewards_good)
        # axes1[1].plot(rewards_adv)

        # # Plotting the epsilon decay
        # plt.figure()
        # plt.plot(epsilon_plotting)
        # plt.title('Epsilon Decay')
        # plt.xlabel('Number of episodes')
        # plt.ylabel('Epsilon value')

        # # Plotting the loss at the end of each episode
        # plt.figure()
        # fig2, axes2 = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
        # plt.figure(figsize=(16, 9))
        # fig2.tight_layout()
        # axes2[0].set(title='Good agents loss', xlabel='Number of episodes', ylabel='Loss')
        # axes1[1].set(title='Adversarial agents loss', xlabel='Number of episodes', ylabel='Loss')
        # axes2[0].plot(loss_plotting_good)
        # axes2[1].plot(loss_plotting_adv.numpy)
        # plt.show()

    def render(self):
        env = simple_tag_v2.env(
            num_good=self.n_good,
            num_adversaries=self.n_adv,
            num_obstacles=self.n_obstacles,
            max_cycles=self.max_cycs / 2,
            continuous_actions=self.continuous,
            render_mode='human'
        )
        total__reward_good = 0
        total__reward_adv = 0
        env.reset()
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            if termination or truncation:
                env.reset()
                break
            if agent != 'agent_0':
                action = self.AdvNetActor.policy(torch.from_numpy(env.last()[0]).to(self.device))
                action = action.cpu().detach().numpy()
                total__reward_adv += reward
                action = np.clip(action, 0, 1)
                # print(action)
            else:
                total__reward_good += reward
                action = None if termination or truncation else env.action_space(agent).sample()
            env.step(action)
            env.render()
            time.sleep(0.005)
        env.close()
        print("Total reward Good: ", total__reward_good)
        print("Total reward Adv: ", total__reward_adv)
        # raise NotImplementedError


if __name__ == "__main__":
    test = TagWorld()
    test.train()
    test.render()
