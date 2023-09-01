import pickle
import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import PPO

from exp_4_cluster_env import ClusterEnv
from exp_4_eval_env import EvalEnv
from exp_4_4_strat.exp_4_4_results import DataCollector


def run_drl():
    pickle_path = r"C:\Users\Kyler\Desktop\GitHub\strat_paper\section_4\exp_4_4_strat\exp_4_4_data.pkl"
    data_collector = pickle.load(open(pickle_path, "rb"))
    env = ClusterEnv(capture_radius=1e6, add_fuel_penalty=True, dense_reward=False, drifting=True, capture_reward=True)
    env.state_buffer = data_collector.start_buffer
    tb_log_path = os.path.join("tb_logs", "exp_4_4_logs")
    eval_env = EvalEnv(buffer_type="default", dense_reward=False)
    eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/",
                             log_path="./logs/", eval_freq=int(1e4), n_eval_episodes=100)
    agent = PPO("MlpPolicy", env, tensorboard_log=tb_log_path, verbose=1, ent_coef=0.01)
    agent.learn(total_timesteps=int(2e6), callback=[eval_callback])


def main():
    run_drl()
    return


if __name__ == "__main__":
    main()

