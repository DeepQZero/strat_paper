import numpy as np
import gymnasium as gym
import torch
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use("TKAgg")

import stable_baselines3 as sb3
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env

from breakout_obs_wrapper import CNNObservation
from breakout_reward_wrapper import SparseReward


def test_model(model_name, reward_wrapped):
    env = make_env(render=False, reward_wrapped=reward_wrapped)
    model = A2C.load(model_name)
    state = env.reset()
    state = torch.tensor(state[0])
    while True:
        state = torch.tensor(state[0])
        print(state.shape)
        dist = model.policy.forward(state)
        dist = model.policy.get_distribution(torch.unsqueeze(state, 0))
        log_prob = dist.log_prob(torch.tensor(list(range(4))))
        prob = torch.exp(log_prob)
        print(prob)
        action, _states = model.predict(state)
        state, reward, done, info = env.step(action)
        # print(state, action, done)


def learn_model(model_name, reward_wrapped=False):
    env = make_env(reward_wrapped=reward_wrapped)
    env.reset()
    assert check_env(env) is None # check if custom environment is suitable for sb3
    tb_log_path = r"./" + model_name + r"_tensorboard/"
    model = A2C("CnnPolicy", env, verbose=1, device="cpu", tensorboard_log=tb_log_path)
    model.learn(total_timesteps=int(1e6))
    model.save(model_name)


def make_env(env_name="ALE/Breakout-v5", render=False, reward_wrapped=False):
    if render:
        env = gym.make(env_name, render_mode="human", obs_type="grayscale")
    else:
        env = gym.make(env_name, obs_type="grayscale")
    env = CNNObservation(env)
    if reward_wrapped:
        env = SparseReward(env)
    #env = gym.wrappers.normalize.NormalizeObservation(env)
    #env = sb3.common.atari_wrappers.AtariWrapper(env)
    return env



def main():
    model_name = "testing"
    print(model_name)
    reward_wrapped = False
    learn_model(model_name, reward_wrapped=reward_wrapped)
    #test_model(model_name, reward_wrapped=reward_wrapped)
    return


if __name__ == "__main__":
    main()
