import numpy as np
import gym
import torch
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TKAgg")
import stable_baselines3 as sb3
from stable_baselines3 import A2C

from breakout_reward_wrapper import SparseReward


def test_model(model_name):
    env = make_env(render=False)
    model = A2C.load(model_name)
    state = env.reset()
    state = torch.tensor(state[0])
    plt.imshow(state)
    plt.show()
    return
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


def learn_model(model_name, wrapped=False):
    env = make_env(wrapped=wrapped)
    env.reset()
    model = A2C("CnnPolicy", env, verbose=1, device="cpu")
    model.learn(total_timesteps=int(1e6))
    model.save(model_name)


def make_env(env_name="ALE/Breakout-v5", render=False, wrapped=False):
    if render:
        env = gym.make(env_name, render_mode="human", obs_type="grayscale")
    else:
        env = gym.make(env_name, obs_type="grayscale")
    env = gym.wrappers.normalize.NormalizeObservation(env)
    #env = sb3.common.atari_wrappers.AtariWrapper(env)
    if wrapped:
        env = SparseReward(env)
    return env


def main():
    model_name = "breakout_A2C_no_wrapper"
    print(model_name)
    #learn_model(model_name, wrapped=False)
    test_model(model_name)
    #model_name = "breakout_A2C_wrapper"
    #print(model_name)
    #learn_model(model_name, wrapped=True)
    #test_model(model_name)
    return


if __name__ == "__main__":
    main()
