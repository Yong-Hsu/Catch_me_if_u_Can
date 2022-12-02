# Catch_me_if_u_Can
DTU 02456 deep learning group project on RL (simple tag and pursuit game)

ddl project poster: Dec 8th around five weeks

1. set up env, try DQN on this
2. Know more about watcher/listener/communication task in this task or cooperative Reinforcement learning
3. if last week runs well, scale this to the pursuit task
4. scale and tweak everything
5. future work and wrap-up

network list:
1. DQN
2. Deep-Deterministic Policy Gradient Algorithm [ddpg](https://spinningup.openai.com/en/latest/algorithms/ddpg.html#:~:text=Deep%20Deterministic%20Policy%20Gradient%20(DDPG,function%20to%20learn%20the%20policy)

3. [Cooperative Multi-agent Control Using Deep Reinforcement Learning](https://link.springer.com/chapter/10.1007/978-3-319-71682-4_5)
4. [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275)


In the pettingzoo env, there should be 

```shell
module unload python # depends
module load python3/3.8.11 #alternatively use 'swap' instead of load'
module load torch 
pip install pettingzoo==1.22.1
```