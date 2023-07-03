import numpy as np
import gymnasium as gym
import torch
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use("TKAgg")

import stable_baselines3 as sb3
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.policies import obs_as_tensor

from breakout_obs_wrapper import CNNObservation
from breakout_reward_wrapper import SparseReward

def predict_action_prob(model, state):
    # TODO: Get working
    # tensor_state = torch.tensor(state).permute(2, 1, 0)
    # dist = model.policy.forward(tensor_state)
    # dist = model.policy.get_distribution(tensor_state)
    # log_prob = dist.log_prob(torch.tensor(list(range(4))))
    # prob = torch.exp(log_prob)
    # print(prob)
    obs = obs_as_tensor(state, model.policy.device)[:, :, 0]
    obs = obs.unsqueeze(dim=1)
    print(obs.shape)
    dist = model.policy.get_distribution(obs)
    probs = dist.distribution.probs
    probs = probs.detach().numpy()
    return probs

def test_model(model_name, reward_wrapped, num_episodes=int(1e1)):
    env = make_env(render=True, reward_wrapped=reward_wrapped)
    model = A2C.load(model_name)
    state = env.reset()
    # Only use the image for forward propagation in model. No access to other state data (eg num lives)
    state = state[0]
    episodes_completed = 0
    while episodes_completed < num_episodes:
        action, _states = model.predict(state)
        state, reward, done1, done2, info = env.step(action)
        #print(predict_action_prob(model, state)) # Not yet working
        if done1 or done2:
            episodes_completed += 1
            state = env.reset()
            state = state[0]



def learn_model(model_name, num_timesteps=int(1e6), reward_wrapped=False):
    env = make_env(reward_wrapped=reward_wrapped)
    env.reset()
    tb_log_path = r"./" + model_name + r"_tensorboard/"
    model = A2C("CnnPolicy", env, verbose=1, device="cpu", tensorboard_log=tb_log_path)
    model.learn(total_timesteps=num_timesteps)
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
    #assert check_env(env) is None # check if custom environment is suitable for sb3
    return env



def main():
    model_name = "breakout_A2C_no_wrapper"
    print(model_name)
    reward_wrapped = False
    #learn_model(model_name, num_timesteps=int(1e3), reward_wrapped=reward_wrapped)
    # tensorboard --logdir ./model_name_tensorboard/
    test_model(model_name, reward_wrapped=reward_wrapped, num_episodes=int(1e1))
    return


if __name__ == "__main__":
    main()
