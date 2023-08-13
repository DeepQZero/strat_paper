from multiprocessing import Pool
import random
import os
import pickle

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import gymnasium as gym
import stable_baselines3
from stable_baselines3 import PPO

from exp_4_2_temp_wrapper import DataCollector
from exp_4_2_env import Env
from exp_4_2_env_wrapper import ClusterEnv
from lib import dynamics as dyn


def fill_buffer():
    data_collector = DataCollector()
    env = Env()
    cluster_env = ClusterEnv()
    i = 0
    while len(data_collector.capture_buffer) < 5:
    #while len(data_collector.start_buffer) < cluster_env.NUM_CLUSTERS:
        if (i % 100) == 0:
            print(i)
        if np.random.uniform(0, 1) < 0.5:
            state = data_collector.buffer_sample()
            if state is None:
                state, _ = env.reset()
            else:
                env.det_reset_helper(state)
        else:
            state, _ = env.reset()
        done = False
        while not done:
            rand_act = data_collector.choose_action(state)
            state, reward, done, _, info = env.step(rand_act)
            data_collector.filter_state(state)
            data_collector.current_trajectory.append(state)
            if env.is_capture():
                data_collector.capture_buffer.append(data_collector.current_trajectory)
        data_collector.current_trajectory = []
        i += 1
    pickle.dump(data_collector, open("trajectory_buffer.pkl", "wb"))

def run_drl():
    data_collector = pickle.load(open("trajectory_buffer.pkl", "rb"))
    env = ClusterEnv()
    env.state_buffer = data_collector.start_buffer
    tb_log_path = os.path.join("tb_logs", "PPO_Cluster_Fill_Buffer_First")
    agent = PPO("MlpPolicy", env, tensorboard_log=tb_log_path, verbose=1, ent_coef=0.01)
    agent.learn(total_timesteps=int(2e6))


if __name__ == "__main__":
    #fill_buffer()
    run_drl()

