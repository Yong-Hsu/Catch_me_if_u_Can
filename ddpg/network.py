import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ActorNet(nn.Module):
    
    def __init__(self, d_in,d_out):
        super(ActorNet,self).__init__()     
        self.action_vector = nn.Sequential(nn.Linear(d_in,32),
                                 nn.ReLU(),
                                 nn.Linear(32,d_out),
                                 nn.Tanh())
              
    def forward(self, x):
        return self.action_vector(x)



class CriticNet(nn.Module):
    
    def __init__(self, d_in,d_out):
        super(CriticNet,self).__init__()     
        self.value = nn.Sequential(nn.Linear(d_in+d_out,32),
                                 nn.ReLU(),
                                 nn.Linear(32,1))
        
    def forward(self, x):
     
        return self.value(x)


class Actor():

    def __init__(self, n_o, n_a):
        self.dim_a = n_a
        self.dim_o = n_o
        self.policy = ActorNet(n_o, n_a)

    def get_action(self, state):
        return self.policy(state)


class Critic():
    
    def __init__(self, n_o, n_a):
        self.dim_o = n_o
        self.value_func = CriticNet(n_o, n_a)
        
    
    def get_state_value(self, state, action):
        x = torch.cat((state,action), dim = 1)
        state_value = self.value_func.forward(x)
        return state_value
