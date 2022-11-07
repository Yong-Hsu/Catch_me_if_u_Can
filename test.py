from pettingzoo.mpe import simple_tag_v2
from gymnasium.utils.save_video import save_video

import time


class Tag:
    def __init__(self, render=False):
        env = simple_tag_v2.env(num_good=1,
                                num_adversaries=3,
                                num_obstacles=2,
                                max_cycles=100,
                                continuous_actions=True,
                                render_mode="human"
                                )

        env.reset()
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            action = None if termination or truncation else env.action_space(
                agent).sample()  # this is where you would insert your policy

            env.step(action)
            time.sleep(0.05)

        env.render()
        env.close()


if __name__ == "__main__":
    test = Tag(render=True)

