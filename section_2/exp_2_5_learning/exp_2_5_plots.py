import gymnasium as gym
import seaborn as sns
import numpy as np


class ActionDist(gym.Env): # WORKING
    def __init__(self, env):
        self.action_uses = [0, 0, 0, 0]
        self.num_steps = 0
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset_action_uses(self):
        self.action_uses = [0, 0, 0, 0]

    def step(self, action):
        self.num_steps += 1
        self.action_uses[int(action)] += 1
        return self.env.step(action)

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

def stacked_density_plot(action_data):
