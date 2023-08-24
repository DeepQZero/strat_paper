# Stacked density plots - discrete actions
import pickle

import numpy as np
import seaborn as sns
import gymnasium as gym


class ActionDist(gym.Env): # WORKING
    def __init__(self, env):
        self.action_uses = [0, 0, 0, 0]
        self.result_hist = []
        self.num_steps = 0
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.PICKLE_NAME = "exp_2_6_data.pkl"

    def step(self, action):
        self.num_steps += 1
        self.action_uses[int(action)] += 1
        if (self.num_steps % int(5e4)) == 0:
            self.result_hist.append([self.num_steps, (np.array(self.action_uses) / np.sum(self.action_uses))])
            self.action_uses = [0, 0, 0, 0]
        if (self.num_steps % int(5e5)) == 0:
            pickle.dump(self.result_hist, open(self.PICKLE_NAME, "wb"))
        return self.env.step(action)

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

def stacked_density_plot(action_data):
    return