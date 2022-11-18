from pettingzoo.mpe import simple_tag_v2
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

        # experience
        self.env.agents[0].replay = [1, 2, 3]
        #

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
        raise NotImplementedError


if __name__ == "__main__":
    test = TagWorld()
    print(test.env.agents[0].replay)
