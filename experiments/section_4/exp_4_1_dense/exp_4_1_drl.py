import time

import numpy as np
import gymnasium as gym
import stable_baselines3 as sb3
from stable_baselines3 import PPO

from lib import dynamics as dyn
from exp_4_1_env import Env

def make_env():
    env = Env()
    return env

def train_agent():
    env = make_env()
    agent = PPO("MlpPolicy", env, verbose=1)
    agent.learn(total_timesteps=int(1e7))

def main():
    train_agent()
    return

if __name__ == "__main__":
    main()