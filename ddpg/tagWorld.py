from pettingzoo.mpe import simple_tag_v2

import copy
import random
from utils import ReplayBuffer, extract_data
from tqdm import tqdm

# import torch
import numpy as np
from network import *


class TagWorld:
    def __init__(self):
        self.env = simple_tag_v2.env(
            # todo: works with one good agent now
            num_good=1,
            num_adversaries=3,
            num_obstacles=2,
            max_cycles=1000,
            continuous_actions=True
        )

        # parameters
        self.GAMMA = 0.99  # Discount factor
        self.ALPHA = 1e-3  # learning rate
        self.TAU = 0.001
        self.BATCH_SIZE = 16
        self.BUFFER_SIZE = 1000
        self.env.reset()
        self.epsilon = 0.9
        self.decay = 0.99

        n_inputs_good = 14
        n_inputs_adv = 16
        n_outputs = 5

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # initiate the networks for the agents
        # the reason why critic network use 1 as output and the unsqueeze later
        self.GoodNetActor = Actor(n_inputs_good, n_outputs)
        self.GoodNetCritic = Critic(n_inputs_good, 1)

        self.AdvNetActor = Actor(n_inputs_adv, n_outputs)
        self.AdvNetCritic = Critic(n_inputs_adv, 1)

        # Target Network
        self.GoodNetActorTarget = copy.deepcopy(self.GoodNetActor)
        self.GoodNetCriticTarget = copy.deepcopy(self.GoodNetCritic)

        self.AdvNetActorTarget = copy.deepcopy(self.AdvNetActor)
        self.AdvNetCriticTarget = copy.deepcopy(self.AdvNetCritic)

        # Experience
        self.ReplayBufferGood = ReplayBuffer(self.BUFFER_SIZE)
        self.ReplayBufferAdv = ReplayBuffer(self.BUFFER_SIZE)

        # optimizer todo:
        self.optim = torch.optim.Adam(self.GoodNetCritic.value_func.parameters(), lr=self.ALPHA)
        self.optim = torch.optim.Adam(self.AdvNetCritic.value_func.parameters(), lr=self.ALPHA)

        self.env.reset()

    # def random_sample(self):
    #     self.env.reset()
    #     for agent in self.env.agent_iter():
    #         observation, reward, termination, truncation, info = self.env.last()
    #         action = None if termination or truncation else self.env.action_space(
    #             agent).sample()  # this is where you would insert your policy
    #
    #         self.env.step(action)
    #
    #     self.env.close()

    def train(self):
        self.env.reset()
        max_episodes = 200
        max_rollout = 500

        for _ in tqdm(range(max_episodes)):
            self.env.reset()
            count = 0

            for agent in self.env.agent_iter():
                # action from network
                observation, _, _, _, _ = self.env.last()
                # todo: add noise
                if agent == 'agent_0':
                    action = self.GoodNetActor.get_action(torch.from_numpy(observation))
                    # tensor([0.1313, -0.0689, -0.0344, 0.1128, -0.3169], grad_fn= < TanhBackward0 >)
                    action = action.detach().numpy()
                    action = np.clip(action, 0, 1)  # clip negative and bigger than 1 values
                else:
                    action = self.AdvNetActor.get_action(torch.from_numpy(observation))
                    action = action.detach().numpy()
                    action = np.clip(action, 0, 1)

                # epsilon greedy, if true, replace the action above
                p = random.random()
                if p < self.epsilon:
                    action = self.env.action_space(agent).sample()
                # Decay greedy epsilon
                self.epsilon = self.epsilon * self.decay

                # Get the new state, reward, and done signal
                self.env.step(action)
                _, reward_new, termination_new, truncation_new, _ = self.env.last()
                observation_new = self.env.observe(agent)

                # store replay buffer
                experience = [observation, action, observation_new, reward_new]
                if agent == 'agent_0':
                    # different iteration or do the calculation
                    self.ReplayBufferGood.append_memory(experience)
                else:
                    self.ReplayBufferAdv.append_memory(experience)

                if termination_new or truncation_new or count > 500:
                    break

                if self.ReplayBufferAdv.buf_len() >= self.BUFFER_SIZE:
                    # todo: should we train adversary and agent at the same time, cause if it is agent turn,
                    # then the adversary network should not be updated? maybe add a if condition
                    # Time to update learnings
                    sampled_experience_good = self.ReplayBufferGood.sample()
                    sampled_experience_adv = self.ReplayBufferAdv.sample()

                    # calculate target network
                    # Good Agent
                    compressed_states_good, compressed_actions_good, compressed_next_states_good, \
                        compressed_rewards_good = extract_data(sampled_experience_good)

                    target_action_good = self.GoodNetActorTarget.get_action(compressed_next_states_good)
                    # todo: should we average the action tensor? Maybe we should use all those info, same below
                    target_action_good = target_action_good.mean(dim=1).unsqueeze(-1)  # (32, 5) -> (32, 1)
                    target_value_good = self.GoodNetCriticTarget.get_state_value(compressed_next_states_good,
                                                                                 target_action_good)

                    # Adversarial Agent
                    compressed_states_adv, compressed_actions_adv, compressed_next_states_adv, compressed_rewards_adv \
                        = extract_data(sampled_experience_adv)

                    target_action_adv = self.AdvNetActorTarget.get_action(compressed_next_states_adv)
                    target_action_adv = target_action_adv.mean(dim=1).unsqueeze(-1)
                    target_value_adv = self.AdvNetCriticTarget.get_state_value(compressed_next_states_adv,
                                                                               target_action_adv)

                    # compute targets
                    target_v_good = compressed_rewards_good.unsqueeze(1) + self.GAMMA * target_value_good
                    target_v_adv = compressed_rewards_adv.unsqueeze(1) + self.GAMMA * target_value_adv

                    # calculate Q function
                    compressed_actions_good = compressed_actions_good.mean(dim=1).unsqueeze(-1)
                    compressed_actions_adv = compressed_actions_adv.mean(dim=1).unsqueeze(-1)
                    actual_v_good = self.GoodNetCritic.get_state_value(compressed_states_good, compressed_actions_good)
                    actual_v_adv = self.AdvNetCritic.get_state_value(compressed_states_adv, compressed_actions_adv)

                    # train the network
                    loss = nn.MSELoss()
                    output_good = loss(actual_v_good, target_v_good)
                    output_adv = loss(actual_v_adv, target_v_adv)

                    self.optim.zero_grad()
                    output_good.backward(retain_graph=True)
                    output_adv.backward(retain_graph=True)
                    self.optim.step()

                    self.GoodNetCritic.value_func.zero_grad()
                    self.AdvNetCritic.value_func.zero_grad()

                    # One step gradient ascent for updating policy
                    for s, a in zip(compressed_states_good.split(1), compressed_actions_good.split(1)):
                        online_v_good = self.GoodNetCritic.get_state_value(s, a)
                        grad_wrt_a_good = torch.autograd.grad(online_v_good, (s, a))

                        action_good = self.GoodNetActor.get_action(s)
                        action_good.mean().backward(retain_graph=True)

                        for param in self.GoodNetActor.policy.parameters():
                            param.data += self.ALPHA * (param.grad * grad_wrt_a_good[1].item()) / self.BATCH_SIZE

                        self.GoodNetActor.policy.zero_grad()
                        self.GoodNetCritic.value_func.zero_grad()

                    for s, a in zip(compressed_states_adv.split(1), compressed_actions_adv.split(1)):
                        online_v_adv = self.AdvNetCritic.get_state_value(s, a)
                        grad_wrt_a_adv = torch.autograd.grad(online_v_adv, (s, a))

                        action_adv = self.AdvNetActor.get_action(s)
                        action_adv.mean().backward(retain_graph=True)

                        for param in self.AdvNetActor.policy.parameters():
                            param.data += self.ALPHA * (param.grad * grad_wrt_a_adv[1].item()) / self.BATCH_SIZE

                        self.AdvNetActor.policy.zero_grad()
                        self.AdvNetCritic.value_func.zero_grad()

                    # Update the target networks
                    # Good agent
                    for param_o_good, param_t_good in zip(self.GoodNetActor.policy.parameters(),
                                                          self.GoodNetActorTarget.policy.parameters()):
                        param_t_good.data = param_o_good.data * self.TAU + param_t_good.data * (1 - self.TAU)

                    for param_o_good, param_t_good in zip(self.GoodNetCritic.value_func.parameters(),
                                                          self.GoodNetCriticTarget.value_func.parameters()):
                        param_t_good.data = param_o_good.data * self.TAU + param_t_good.data * (1 - self.TAU)

                    # Adversarial agent
                    for param_o_adv, param_t_adv in zip(self.AdvNetActor.policy.parameters(),
                                                        self.AdvNetActorTarget.policy.parameters()):
                        param_t_adv.data = param_o_adv.data * self.TAU + param_t_adv.data * (1 - self.TAU)

                    for param_o_adv, param_t_adv in zip(self.AdvNetCritic.value_func.parameters(),
                                                        self.AdvNetCriticTarget.value_func.parameters()):
                        param_t_adv.data = param_o_adv.data * self.TAU + param_t_adv.data * (1 - self.TAU)

                    self.GoodNetActor.policy.zero_grad()
                    self.GoodNetActorTarget.policy.zero_grad()
                    self.GoodNetCritic.value_func.zero_grad()
                    self.GoodNetCriticTarget.value_func.zero_grad()

                    # torch.save(self.GoodNetActorTarget.policy.state_dict(), 'good_target_actor_state_1.pt')
                    # torch.save(self.GoodNetCriticTarget.value_func.state_dict(), 'good_target_critic_state_1.pt')

                    self.AdvNetActor.policy.zero_grad()
                    self.AdvNetActorTarget.policy.zero_grad()
                    self.AdvNetCritic.value_func.zero_grad()
                    self.AdvNetCriticTarget.value_func.zero_grad()
                    # torch.save(self.AdvNetActorTarget.policy.state_dict(), 'adv_target_actor_state_1.pt')
                    # torch.save(self.AdvNetCriticTarget.value_func.state_dict(), 'adv_target_critic_state_1.pt')

                count = count + 1


if __name__ == "__main__":
    test = TagWorld()
    test.train()
