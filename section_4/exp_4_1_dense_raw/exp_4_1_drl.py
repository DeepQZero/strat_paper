import os

import numpy as np
import gymnasium as gym
import stable_baselines3 as sb3
from stable_baselines3 import PPO

from lib import dynamics as dyn
from exp_4_1_env import Env
from capture_callback import CaptureCallback

def make_env():
    env = Env()
    return env

def train_agent():
    #env = Env(add_fuel_penalty=False)
    #tb_log_path = os.path.join("tb_logs", "PPO_testing")
    #agent = PPO("MlpPolicy", env, tensorboard_log=tb_log_path, verbose=1)
    #agent.learn(total_timesteps=int(5e6), callback=CaptureCallback())
    env2 = make_env()
    tb_log_path = os.path.join("tb_logs", "PPO_FuelPen")
    agent = PPO("MlpPolicy", env2, tensorboard_log=tb_log_path, verbose=1, ent_coef=0.01)
    agent.learn(total_timesteps=int(5e6), callback=CaptureCallback())

def main():
    train_agent()
    return

if __name__ == "__main__":
    main()