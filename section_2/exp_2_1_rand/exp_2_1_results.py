from multiprocessing import Pool
import time
import pickle

import numpy as np

from lib import dynamics as dyn
from experiments.section_2.exp_2_1_rand.exp_2_1_env import Env


def one_episode(params):
    """Gets data from one episode."""
    thrust, bound = params
    env = Env()
    _ = env.reset()
    done = False
    is_halfway = False
    capture_turn = 0
    return_turn = 0
    fuel_total = 0
    while not done:
        rand_thrust = np.random.uniform(0, thrust)
        rand_angle = np.random.uniform(0, 2*np.pi)
        rand_act = np.array([np.cos(rand_angle), np.sin(rand_angle)]) * rand_thrust
        fuel_total += dyn.vec_norm(rand_act)
        state, reward, done, info = env.step(rand_act)
        if not is_halfway and (dyn.vec_norm(state[0:2]-state[8:10]) <= bound):
            is_halfway = True
            capture_turn = state[12]
        if is_halfway and (dyn.vec_norm(state[0:2]-state[4:6]) <= bound):
            done = True
            return_turn = state[12]
    return capture_turn, return_turn, fuel_total


def get_data(thrust, bound, episodes):
    """Gets experiment data and returns dictionary of polished data."""
    tic = time.time()
    with Pool(16) as p:
        all_data = p.map(one_episode, [(thrust, bound)]*episodes)
    toc = time.time()
    capture_counter = 0
    return_counter = 0
    capture_delta_v_total = 0
    return_delta_v_total = 0
    for episode_data in all_data:
        if episode_data[0] > 0:
            capture_counter += 1
            capture_delta_v_total += episode_data[2]
        if episode_data[1] > 0:
            return_counter += 1
            return_delta_v_total += episode_data[2]
    print('Time: ', toc-tic, ' Bound: ', bound, ' Thrust: ', thrust)
    return {'episodes': episodes, 'tot_time': toc-tic,
            'bound': bound, 'thrust': thrust,
            'num_captures': capture_counter, 'num_returns': return_counter,
            'total_capture_delta_v': capture_delta_v_total,
            'total_return_delta_v': return_delta_v_total}


def main_exp():
    """"Main experiment function."""
    data_dict = {}
    episodes = int(1e3)
    for thrust in [1, 5, 10, 50, 100]:
        for bound in [1e4, 1e5, 1e6, 1e7]:
            data = get_data(thrust, bound, episodes)
            data_dict[(thrust, bound)] = data
    with open('exp_2_1_data.pkl', 'wb') as f:
        pickle.dump(data_dict, f)


if __name__ == "__main__":
    main_exp()
