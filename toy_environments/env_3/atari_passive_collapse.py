import time
import gc

import numpy as np
import gymnasium as gym
import stable_baselines3.common.vec_env
import torch
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use("TKAgg")

import stable_baselines3 as sb3
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.policies import obs_as_tensor

from breakout_obs_wrapper import CNNObservation
from breakout_reward_wrapper import SparseReward
from wrappers import *
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
    model = DQN.load(model_name, env=env)
    model.device = "cpu"
    state = env.reset()
    # Only use the image for forward propagation in model. No access to other state data (eg num lives)
    state = state[0]
    episodes_completed = 0
    while episodes_completed < num_episodes:
        action, _states = model.predict(state)
        #print(action)
        state, reward, done1, done2, info = env.step(action)
        if done1 or done2:
            episodes_completed += 1
            state = env.reset()
            state = state[0]


def learn_model(model_name, num_timesteps=int(1e6), reward_wrapped=False):
    env = make_env(render=False, reward_wrapped=reward_wrapped)
    tb_log_path = r"./" + model_name + r"_tensorboard/"
    # Normalizes images by default - see docs
    model = DQN("CnnPolicy", env, verbose=1, device="cuda", tensorboard_log=tb_log_path, learning_starts=int(1e5),
                exploration_fraction=0.1, exploration_final_eps=0.01, batch_size=32)
    model.learn(total_timesteps=num_timesteps)
    model.save(model_name)
    del model # delete the model object/replay buffer from RAM. Model file remains stored.
    gc.collect()


def make_env(env_name="ALE/Breakout-v5", render=False, reward_wrapped=False):
    # BreakoutNoFrameskip-v4, ALE/Breakout-v5
    if render:
        env = gym.make(env_name, render_mode="human", repeat_action_probability=0.00,
                       full_action_space=False, frameskip=1) # obs_type="grayscale"
    else:
        env = gym.make(env_name, repeat_action_probability=0.00, full_action_space=False,
                       frameskip=1) # obs_type="grayscale"
    env = EpisodicLifeWrapper(env)
    env = FireResetWrapper(env)
    env = WarpFrameWrapper(env)
    env = FrameStackWrapper(env, k=4) # k = 4
    env = ManyActionWrapper(env, k=4) # k = 4
    env = ClipRewardWrapper(env)
    env = MinimalActionWrapper(env)
    if reward_wrapped:
        env = SparseReward(env)
    #env = CNNObservation(env)
    #assert check_env(env) is None # check if custom environment is suitable for sb3
    return env

def testing():
    env = make_env(render=True, reward_wrapped=False)
    state = env.reset()
    # Only use the image for forward propagation in model. No access to other state data (eg num lives)
    episodes_completed = 0
    while episodes_completed < 10:
        action = np.random.randint(4)
        state, reward, done1, done2, info = env.step(action)
        print(reward)
        if done1 or done2:
            episodes_completed += 1
            state = env.reset()


def main():
    model_name = "breakout_DQN_no_wrapper_1M"
    print(model_name)
    reward_wrapped = False
    learn_model(model_name, num_timesteps=int(1e6), reward_wrapped=False)
    # tensorboard --logdir f./model_name_tensorboard/
    # Evaluate without sparsified rewards?
    #test_model("breakout_DQN_no_wrapper", reward_wrapped=False, num_episodes=int(1e1))
    return

# life loss wrapper - give a reward after life lost, not episode
if __name__ == "__main__":
    #learn_model("breakout_DQN_no_wrapper_2M_1", num_timesteps=int(2e6), reward_wrapped=False)
    #learn_model("breakout_DQN_wrapper_2M_1", num_timesteps=int(2e6), reward_wrapped=True)

    #learn_model("breakout_DQN_no_wrapper_2M_2", num_timesteps=int(2e6), reward_wrapped=False)
    #learn_model("breakout_DQN_wrapper_2M_2", num_timesteps=int(2e6), reward_wrapped=True)

    #learn_model("breakout_DQN_no_wrapper_2M_3", num_timesteps=int(2e6), reward_wrapped=False)
    #learn_model("breakout_DQN_wrapper_2M_3", num_timesteps=int(2e6), reward_wrapped=True)

    #learn_model("breakout_DQN_no_wrapper_2M_4", num_timesteps=int(2e6), reward_wrapped=False)
    #learn_model("breakout_DQN_wrapper_2M_4", num_timesteps=int(2e6), reward_wrapped=True)

    #learn_model("breakout_DQN_no_wrapper_2M_5", num_timesteps=int(2e6), reward_wrapped=False)
    #learn_model("breakout_DQN_wrapper_2M_5", num_timesteps=int(2e6), reward_wrapped=True)
    test_model("breakout_DQN_no_wrapper_4M", reward_wrapped=False, num_episodes=10)
    #learn_model("breakout_DQN_wrapper_1M", num_timesteps=int(1e6), reward_wrapped=True)
    #main()
