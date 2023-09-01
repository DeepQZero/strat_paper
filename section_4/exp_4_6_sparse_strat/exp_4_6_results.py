import pickle
import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import PPO

from exp_4_cluster_env import ClusterEnv
from exp_4_eval_env import EvalEnv
from exp_4_space_env import Env
from exp_4_4_strat.exp_4_4_results import DataCollector


def run_drl():
    pickle_path = "exp_4_6_data.pkl"
    #pickle_path = r"C:\Users\Kyler\Desktop\GitHub\strat_paper\section_4\exp_4_4_strat\exp_4_4_data.pkl"
    data_collector = pickle.load(open(pickle_path, "rb"))
    env = ClusterEnv(capture_radius=1e6, add_fuel_penalty=True, dense_reward=False, drifting=True, capture_reward=True)
    env.state_buffer = data_collector.start_buffer
    tb_log_path = os.path.join("prefill_state_traj/tb_logs", "exp_4_6_logs")
    eval_env = EvalEnv(buffer_type="default", dense_reward=False)
    eval_callback = EvalCallback(eval_env, best_model_save_path="prefill_state_traj/logs/",
                                 log_path="prefill_state_traj/logs/", eval_freq=int(1e4), n_eval_episodes=100)
    agent = PPO("MlpPolicy", env, tensorboard_log=tb_log_path, verbose=1, ent_coef=0.01)
    agent.learn(total_timesteps=int(2e6), callback=[eval_callback])

def fill_capture_buffer():
    data_collector = DataCollector()
    env = Env(capture_radius=5e5)
    i = 0
    while len(data_collector.capture_buffer) < 500:
        if (i % 100) == 0:
            print(i)
        if np.random.uniform(0, 1) < 0.5:
            state = data_collector.buffer_sample()
            if state is None:
                state, _ = env.reset()
                state = env.det_obs_1()
            else:
                for u in data_collector.start_trajectory_buffer:
                    # data_collector.start_trajectory_buffer.pop(u)
                    if np.linalg.norm(u[0] - state) < 1e-6:
                        data_collector.current_trajectory = u[1]
                env.det_reset_helper(state)
        else:
            state, _ = env.reset()
            state = env.det_obs_1()
        done = False
        while not done:
            rand_act = data_collector.choose_action(state)
            state, reward, done, _, info = env.step(rand_act)
            state = env.det_obs_1()
            data_collector.filter_state(state)
            data_collector.current_trajectory.append(state)
            if env.is_capture():
                #print(data_collector.current_trajectory)
                data_collector.start_trajectory_buffer.append([state, data_collector.current_trajectory])
                data_collector.capture_buffer.append(data_collector.current_trajectory)
        data_collector.current_trajectory = []
        i += 1
    pickle.dump(data_collector, open("exp_4_6_data.pkl", "wb"))


def main():
    fill_capture_buffer()
    run_drl()
    return


if __name__ == "__main__":
    main()

