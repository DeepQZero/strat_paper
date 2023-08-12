from multiprocessing import Pool
import random

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import gymnasium as gym

from exp_4_2_temp_wrapper import DataCollector
from exp_4_2_env import Env
from exp_4_2_env_wrapper import ClusterEnv
from lib import dynamics as dyn


def main():
    data_collector = DataCollector()
    env = Env()
    cluster_env = ClusterEnv()
    for i in range(int(1e6)):
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
        if int(len(data_collector.start_buffer)) == int(cluster_env.NUM_CLUSTERS):
            break






if __name__ == "__main__":
    main()