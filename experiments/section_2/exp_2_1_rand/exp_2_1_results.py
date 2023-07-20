from multiprocessing import Pool
import time
import pickle

import numpy as np

from lib import dynamics as dyn
from experiments.section_2.exp_2_1_rand.exp_2_1_env import Env


def one_episode(params):
    thrust, bound = params
    env = Env()
    _ = env.reset()
    done = False
    is_halfway = False
    capture_turn = 0
    return_turn = 0
    fuel_total = 0
    while not done:
        random_act = np.random.uniform(-thrust, thrust, 2)
        fuel_total += dyn.vec_norm(random_act)
        state, reward, done, info = env.step(random_act)
        if not is_halfway and (dyn.vec_norm(state[0:2]-state[8:10]) <= bound):
            is_halfway = True
            capture_turn = state[12]
        if is_halfway and (dyn.vec_norm(state[0:2]-state[4:6]) <= bound):
            done = True
            return_turn = state[12]
    return capture_turn, return_turn, fuel_total


def get_data(thrust, bound, episodes):
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
    return {'times': episodes, 'tot_time': toc-tic,
            'bound': bound, 'upper_thrust': thrust,
            'captures': capture_counter, 'returns': return_counter,
            'capture_delta_v': capture_delta_v_total,
            'return_delta_v': return_delta_v_total}


def main_exp():
    data_dict = {}
    episodes = int(1e2)
    for thrust in [0.5, 1, 2, 5, 10]:
        for bound in [1e4, 1e5, 1e6, 1e7]:
            data = get_data(thrust, bound, episodes)
            data_dict[(thrust, bound)] = data
    with open('exp_2_1_data.pkl', 'wb') as f:
        pickle.dump(data_dict, f)


if __name__ == "__main__":
    main_exp()
