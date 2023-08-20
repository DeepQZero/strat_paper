# TODO: Density plots of actions during training
#       Stacked density plot of the passive action distribution
#       Pickle the data, put it in the dropbox

import gc

import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback

from exp_2_4_env import *


def test_model(model_name, reward_wrapped, num_episodes=int(1e1)):
    env = make_env(render=True, reward_wrapped=reward_wrapped)
    model = DQN.load(model_name, env=env)
    model.device = "cpu"
    state = env.reset()[0]
    episodes_completed = 0
    while episodes_completed < num_episodes:
        action, _states = model.predict(state)
        state, reward, done1, done2, info = env.step(action)
        if done1 or done2:
            episodes_completed += 1
            state = env.reset()[0]


def learn_model(model_name, num_timesteps=int(1e6), reward_wrapped=False):
    env = make_env(render=False, reward_wrapped=reward_wrapped)
    tb_log_path = r"./" + model_name + r"_tensorboard/"
    # Normalizes images by default - see docs
    model = DQN("CnnPolicy", env, verbose=1, device="cuda", tensorboard_log=tb_log_path, learning_starts=int(1e5),
                exploration_fraction=0.1, exploration_final_eps=0.01, batch_size=32)
    eval_callback = EvalCallback() # TODO FILL IN ARGS
    model.learn(total_timesteps=num_timesteps, callback=[eval_callback])
    model.save(model_name)
    del model # delete the model object/replay buffer from RAM. Model file remains stored.
    gc.collect() # Clear out the RAM after training is done so models can be trained in succession.


def make_env(env_name="ALE/Breakout-v5", render=False, reward_wrapped=False):
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
    #learn_model("breakout_DQN_no_wrapper_2M_1", num_timesteps=int(2e6), reward_wrapped=False)
    #learn_model("breakout_DQN_wrapper_2M_1", num_timesteps=int(2e6), reward_wrapped=True)

    # learn_model("breakout_DQN_no_wrapper_2M_2", num_timesteps=int(2e6), reward_wrapped=False)
    # learn_model("breakout_DQN_wrapper_2M_2", num_timesteps=int(2e6), reward_wrapped=True)

    # learn_model("breakout_DQN_no_wrapper_2M_3", num_timesteps=int(2e6), reward_wrapped=False)
    # learn_model("breakout_DQN_wrapper_2M_3", num_timesteps=int(2e6), reward_wrapped=True)

    # learn_model("breakout_DQN_no_wrapper_2M_4", num_timesteps=int(2e6), reward_wrapped=False)
    # learn_model("breakout_DQN_wrapper_2M_4", num_timesteps=int(2e6), reward_wrapped=True)

    # learn_model("breakout_DQN_no_wrapper_2M_5", num_timesteps=int(2e6), reward_wrapped=False)
    # learn_model("breakout_DQN_wrapper_2M_5", num_timesteps=int(2e6), reward_wrapped=True)
    # tensorboard --logdir f./model_name_tensorboard/
    #test_model("breakout_DQN_no_wrapper", reward_wrapped=False, num_episodes=int(1e1))
    return

if __name__ == "__main__":
    main()
