<<<<<<< HEAD
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
from tagWorld import *
import torch



model = ActorNet(16,5,32)
model.load_state_dict(torch.load(r'C:\Users\albor\Desktop\CMIYC pre trained\AdvNetActor_1669992271.3201299.pt', map_location=torch.device('cpu')), strict=False)
=======
import torch
from pettingzoo.mpe import simple_tag_v2


# Model class must be defined somewhere
model = torch.jit.load(r'"E:\DTU\DeepLearning\Catch_me_if_u_Can\ddpg\AdvNetActor_1670158469.7885914.pt"')
>>>>>>> b32e97332c23deed4ba170c7c76ba64e3b069f8d
model.eval()




def render():
    env = simple_tag_v2.env(
        num_good=1,
        num_adversaries=3,
        num_obstacles=2,
        max_cycles=3000,
        continuous_actions=True,
        render_mode='human'
    )

    env.reset()
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        if agent != 'agent_0':
            if termination or truncation:
                env.reset()
                continue
            action = model(torch.from_numpy(env.last()[0]))
            action = action.cpu().detach().numpy()
            action = np.clip(action, 0, 1)
            # print(action)
        else:
            action = None if termination or truncation else env.action_space(agent).sample()

        env.step(action)
        env.render()
        time.sleep(0.01)
    env.close()
    # raise NotImplementedError


render()