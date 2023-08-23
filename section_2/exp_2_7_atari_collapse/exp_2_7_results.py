# TODO: Density plots of actions during training
#       Stacked density plot of the passive action distribution
#       Pickle the data, put it in the dropbox

import gc

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback

from exp_2_7_env import make_env

def learn_model(model_name, num_timesteps=int(1e6), reward_wrapped=False):
    env = make_env(render=False, reward_wrapped=reward_wrapped)
    tb_log_path = r"./" + model_name + r"_tensorboard/"
    # Normalizes images by default - see docs
    model = DQN("CnnPolicy", env, verbose=1, device="cuda", tensorboard_log=tb_log_path, learning_starts=int(1e5),
                exploration_fraction=0.1, exploration_final_eps=0.01, batch_size=32)
    eval_env = make_env(render=False, reward_wrapped=False)
    eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/",
                             log_path="./logs/", eval_freq=int(1e4), n_eval_episodes=100)
    model.learn(total_timesteps=num_timesteps, callback=[eval_callback])
    model.save(model_name)
    del model # delete the model object/replay buffer from RAM. Model file remains stored.
    gc.collect() # Clear out the RAM after training is done so models can be trained in succession.

def main():
    learn_model("breakout_DQN_no_wrapper_2M_1", num_timesteps=int(2e6), reward_wrapped=False)
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
