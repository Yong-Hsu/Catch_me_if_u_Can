from pettingzoo.mpe import simple_tag_v2

env = simple_tag_v2.env(num_good=1,
                        num_adversaries=3,
                        num_obstacles=2,
                        max_cycles=50000,
                        continuous_actions=True,
                        render_mode="human"
                        )

env.reset()
i = 0
for agent in env.agent_iter():

    print(agent, ':')
    print(env.observe(agent))
    # print(env.last())
    # print(env.rewards[agent])
    env.step(env.action_space(agent).sample())
    # observation space: shape(14),(16) Agent and adversary observations: [self_vel, self_pos,
    # landmark_rel_positions, other_agent_rel_positions, other_agent_velocities] observation function in the
    # simple_tag.py: other_agent_rel_positions contains all agents except oneself, the order is adversary to agent,
    # other_agent_velocities only for not adversary

    i = i + 1
    if i == 4:
        break

env.close()
