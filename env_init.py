# basic test ground 

from pettingzoo.mpe import simple_tag_v2
env = simple_tag_v2.env(num_good=10, num_adversaries=3, num_obstacles=2, max_cycles=100, continuous_actions=False, render_mode="human")

env.reset()
# while True:
#     for agent in env.agent_iter():
#         observation, reward, termination, truncation, info = env.last()
#         action = None if termination or truncation else env.action_space(agent).sample()  # this is where you would insert your policy
#         env.step(action)
#     if termination == True:
#         break

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    print(termination)
    action = None if termination or truncation else env.action_space(agent).sample()  # this is where you would insert your policy
    env.step(action)

env.render()
env.close()