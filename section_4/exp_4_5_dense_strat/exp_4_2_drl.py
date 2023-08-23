import os

import numpy as np
import gymnasium as gym
import stable_baselines3 as sb3
from stable_baselines3 import PPO

from lib import dynamics as dyn
from exp_4_2_env_wrapper import ClusterEnv
#from sb3_callback import CaptureCallback

def make_env():
    env = ClusterEnv()
    return env

def train_agent():
    env = ClusterEnv()
    tb_log_path = os.path.join("tb_logs", "PPO_clustering")
    agent = PPO("MlpPolicy", env, tensorboard_log=tb_log_path, verbose=1, ent_coef=0.01)
    agent.learn(total_timesteps=int(5e6))

def main():
    train_agent()
    return

if __name__ == "__main__":
    main()