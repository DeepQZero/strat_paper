from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import A2C
from stable_baselines3 import DQN
from stable_baselines3 import TD3

from environments.env_1.racetrack_cont import Env

import gym


# Parallel environments
env = Env()
model = PPO("MlpPolicy", env, ent_coef=0.1, verbose=1, device='cpu')
model.learn(total_timesteps=int(1e7))
