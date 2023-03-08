from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from environments.env_1.racetrack import Env

import gym


# Parallel environments
env = Env()

model = PPO("MlpPolicy", env, ent_coef=0.1, verbose=0)
model.learn(total_timesteps=int(1e7))
