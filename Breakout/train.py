import gym
from time import sleep

env = gym.make('Breakout-v0')
state = env.reset()

for _ in range(100):
    env.render()
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    sleep(0.1)
    print('lala')
    if done:
        state = env.reset()
env.close()
