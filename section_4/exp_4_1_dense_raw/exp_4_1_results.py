import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from exp_4_space_env import Env

def make_env(add_fuel_penalty=True, dense_reward=True):
    env = Env(add_fuel_penalty=add_fuel_penalty, dense_reward=dense_reward)
    return env

def train_agent():
    env = make_env(add_fuel_penalty=False, dense_reward=True)
    eval_env = make_env(add_fuel_penalty=True, dense_reward=True) # So evaluation is consistent, give fuel penalty during evaluation
    tb_log_path = os.path.join("tb_logs", "exp_4_1")
    agent = PPO("MlpPolicy", env, tensorboard_log=tb_log_path, verbose=1, ent_coef=0.01)
    eval_callback = EvalCallback(eval_env, n_eval_episodes=100, eval_freq=int(2e4))
    agent.learn(total_timesteps=int(1e6), callback=[eval_callback])

def main():
    train_agent()
    return

if __name__ == "__main__":
    main()