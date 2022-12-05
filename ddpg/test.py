import unittest
from pettingzoo.mpe import simple_tag_v2
from network import Actor, Critic
import torch


# test class for testing the functionality of the network
class TestStringMethods(unittest.TestCase):

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_actor(self):
        goodActor = Actor(14, 5)
        advActor = Actor(16, 5)

        env = simple_tag_v2.env(
            num_good=1,
            num_adversaries=3,
            num_obstacles=2,
            max_cycles=8,
            continuous_actions=True
        )

        env.reset()
        for agent in env.agent_iter():
            observation, _, termination, truncation, _ = env.last()
            action = None if termination or truncation else env.action_space(
                agent).sample()  # this is where you would insert your policy

            if agent == 'agent_0':
                res = goodActor.get_action(torch.from_numpy(observation))
                self.assertTrue(res.shape[0] == 5)
            else:
                res = advActor.get_action(torch.from_numpy(observation))

            env.step(action)

        env.close()

    def render_pt(self):
        raise NotImplementedError


if __name__ == '__main__':
    unittest.main()
