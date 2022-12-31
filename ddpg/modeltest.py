from tagWorld import *
import torch
from pettingzoo.mpe import simple_tag_v2

model = ActorNet(12, 5, 32)
model.load_state_dict(torch.load(r'AdvNetActor_23_2.pt',
                      map_location=torch.device('cuda')),
                      strict=False)


def render():
    env = simple_tag_v2.env(
        num_good=1,
        num_adversaries=3,
        num_obstacles=0,
        max_cycles=150,
        continuous_actions=True,
        render_mode='human'
    )

    env.reset()
    dist = []
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        if agent != 'agent_0':
            if termination or truncation:
                env.reset()
                print(sum(dist) / len(dist))
                dist = []
                continue
            # action = model(torch.from_numpy(env.last()[0]))
            # action = action.cpu().detach().numpy()

            action = None if termination or truncation else env.action_space(agent).sample()
            dist.append(np.linalg.norm((observation[8], observation[9])))
        else:
            action = None if termination or truncation else env.action_space(agent).sample()

        # time.sleep(0.01)
        env.step(action)

    env.render()
    env.close()


render()
