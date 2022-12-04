import torch
from pettingzoo.mpe import simple_tag_v2


# Model class must be defined somewhere
model = torch.jit.load(r'"E:\DTU\DeepLearning\Catch_me_if_u_Can\ddpg\AdvNetActor_1670158469.7885914.pt"')
model.eval()

def render(self):
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
            action = self.AdvNetActor.policy(torch.from_numpy(env.last()[0]).to(self.device))
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
