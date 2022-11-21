from pettingzoo.mpe import simple_tag_v2
from collections import deque
import random
from network import *
import copy
from tqdm import tqdm
# from gymnasium.utils.save_video import save_video


class TagWorld:
    def __init__(self):
        self.env = simple_tag_v2.env(
            num_good=1,
            num_adversaries=3,
            num_obstacles=2,
            max_cycles=1000,
            continuous_actions=False
        )

        
        self.env.reset()
        self.epsilon = 0.9
        self.decay = 0.99
        n_inputs_good = 14
        n_inputs_adv = 8
        n_outputs = 5

        self.GoodNetActor = Actor(n_inputs_good,n_outputs)
        self.GoodNetCritic = Critic(n_inputs_good, n_outputs)

        self.AdvNetActor = Actor(n_inputs_adv, n_outputs)
        self.AdvNetCritic = Critic(n_inputs_adv,n_outputs)

        # Target Network
        self.GoodNetActorTarget = copy.deepcopy(self.GoodNetActor)
        self.GoodNetCriticTarget = copy.deepcopy(self.GoodNetCritic)

        self.AdvNetActorTarget = copy.deepcopy(self.AdvNetActor)
        self.AdvNetCriticTarget = copy.deepcopy(self.AdvNetCritic)

        # Experience
        self.ReplayBufferGood = ReplayBuffer()
        self.ReplayBufferAdv = ReplayBuffer()

        self.env.reset()


    def random_sample(self):
        self.env.reset()
        for agent in self.env.agent_iter():
            observation, reward, termination, truncation, info = self.env.last()
            action = None if termination or truncation else self.env.action_space(
                agent).sample()  # this is where you would insert your policy

            self.env.step(action)

        self.env.close()


    def train(self):
        # raise NotImplementedError

        self.env.reset()
        max_episodes = 4500
        for i in tqdm(range(max_episodes)):
            self.env.reset()
            for agent in self.env.agent_iter():
                observation, reward, termination, truncation, info = self.env.last()
                if agent != 'agent_0':
                    action = self.GoodNetActor.get_action(observation)
                else:
                    action = self.AdvNetActor.get_action(observation)
                
                p = random.random()
                if p < self.epsilon:
                    action = self.env.action_space(agent).sample()
                
                # Decay greedy epsilon
                self.epsilon = self.epsilon * self.decay

                # Get the new state, reward, and done signal
                observation_new, reward_new, termination_new, truncation_new, info_new = self.env.step(action)
                experience = [observation, action, reward_new, termination_new]

                if agent != 'agent_0':
                    self.ReplayBufferGood.append_memory(experience)
                else:
                    self.ReplayBufferAdv.append_memory(experience)
                
                if termination_new is True:
                    break
            
            # Time to update learnings

            for 
                    








                
                






if __name__ == "__main__":
    test = TagWorld()
    print(test.env.agents[0].replay)




class ReplayBuffer:
    def __init__(self, max_length):
        self.max_length = max_length
        self.Buffer = deque(maxlen = self.max_length) #The last firt entry gets automatically removed when the buffer size is exceeded

    def append_memory(self, experience):
        self.Buffer.appendleft(experience)
    
    def sample(self. batch_size = 32):
        return random.sample(self.Buffer, batch_size)
