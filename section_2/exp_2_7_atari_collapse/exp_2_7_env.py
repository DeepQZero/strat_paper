import os
import sys

import gymnasium as gym

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from exp_2_6_atari_learning import exp_2_6_env

class PenalizeAction(gym.Wrapper):
    def __init__(self, env):
        super(PenalizeAction, self).__init__(env)

    def step(self, action):
        obs, rew, done1, done2, info = self.env.step(action)
        if action == 2 or action == 3:
            rew = rew - 1.0
        return obs, rew, done1, done2, info

def make_env(env_name="ALE/Breakout-v5", render=False, reward_wrapped=True):
    env = exp_2_6_env.make_env(env_name=env_name, render=render, reward_wrapped=reward_wrapped)
    env.PICKLE_NAME = "exp_2_7_data.pkl"
    env = PenalizeAction(env)
    return env
