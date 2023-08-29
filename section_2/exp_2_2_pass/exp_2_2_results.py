from multiprocessing import Pool
import time
import pickle
import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np

from lib import dynamics as dyn
from exp_2_1_rand.exp_2_1_env import BaseSpaceEnv


def one_episode(params):
    """Gets data from one episode."""
    thrust, bound, passive_prop = params
    env = BaseSpaceEnv()
    _ = env.reset()
    done = False
    rendez_turn = 0
    fuel_total = 0
    while not done:
        if np.random.uniform(0, 1) < passive_prop:
            action = np.array([0.0, 0.0])
        else:
            rand_thrust = np.random.uniform(0, thrust)
            rand_angle = np.random.uniform(0, 2 * np.pi)
            action = np.array([np.cos(rand_angle), np.sin(rand_angle)]) * rand_thrust
        fuel_total += dyn.vec_norm(action)
        state, reward, done, info = env.step(action)
        if dyn.vec_norm(state[0:2]-state[4:6]) <= bound:
            done = True
            rendez_turn = state[8]
    return rendez_turn, fuel_total


def get_data(thrust, bound, passive_prop, episodes):
    """Gets experiment data and returns dictionary of polished data."""
    tic = time.time()
    with Pool(16) as p:
        all_data = p.map(one_episode, [(thrust, bound, passive_prop)]*episodes)
    toc = time.time()
    rendez_counter = 0
    rendez_delta_v_total = 0
    for episode_data in all_data:
        if episode_data[0] > 0:
            rendez_counter += 1
            rendez_delta_v_total += episode_data[1]
    print('Time: ', toc-tic, ' Passive Prop: ', passive_prop,
          ' Thrust: ', thrust)
    return {'episodes': episodes, 'tot_time': toc-tic,
            'bound': bound, 'thrust': thrust,
            'num_rendez': rendez_counter,
            'total_rendez_delta_v': rendez_delta_v_total,
            'passive_prop': passive_prop}


def main_exp():
    """"Main experiment function."""
    data_dict = {}
    episodes = int(1e5)
    bound = 1e6
    for thrust in [5, 7.5, 10, 15, 20]:
        for passive_prop in [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
            data = get_data(thrust, bound, passive_prop, episodes)
            data_dict[(thrust, passive_prop)] = data
    with open('exp_2_2_data.pkl', 'wb') as f:
        pickle.dump(data_dict, f)


if __name__ == "__main__":
    main_exp()
