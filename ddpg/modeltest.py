from tagWorld import *
import torch
from pettingzoo.mpe import simple_tag_v2

model = ActorNet(12, 5, 32)
model.load_state_dict(torch.load(r'AdvNetActor_1670336662.502053.pt',
                      map_location=torch.device('cuda')),
                      strict=False)


def render():
    env = simple_tag_v2.env(
        num_good=1,
        num_adversaries=3,
        num_obstacles=0,
        max_cycles=3000,
        continuous_actions=True,
        render_mode='human'
    )

    env.reset()
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        if agent != 'agent_0':
            if termination or truncation:
                print(1)
                env.reset()
                continue
            action = model(torch.from_numpy(env.last()[0]))
            action = action.cpu().detach().numpy()
            action = (np.clip(action, -1, 1) + 1) / 2
            print(action)
            print(env.observe(agent)[0], env.observe(agent)[1])
            print('-------------------------------------')
        else:
            action = None if termination or truncation else env.action_space(agent).sample()

        env.step(action)
        env.render()
        time.sleep(0.01)
    env.close()


render()
