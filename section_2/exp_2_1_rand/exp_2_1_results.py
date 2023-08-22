from multiprocessing import Pool
import time
import pickle

import numpy as np

from lib import dynamics as dyn
from experiments.section_2.exp_2_1_rand.exp_2_1_env import BaseSpaceEnv


def one_episode(params):
    """Gets data from one episode."""
    thrust, bound = params
    env = BaseSpaceEnv()
    _ = env.reset()
    done = False
    rendez_turn = 0
    fuel_total = 0
    while not done:
        rand_thrust = np.random.uniform(0, thrust)
        rand_angle = np.random.uniform(0, 2*np.pi)
        rand_act = np.array([np.cos(rand_angle), np.sin(rand_angle)]) * rand_thrust
        fuel_total += dyn.vec_norm(rand_act)
        state, reward, done, info = env.step(rand_act)
        if dyn.vec_norm(state[0:2]-state[4:6]) <= bound:
            done = True
            rendez_turn = state[8]
    return rendez_turn, fuel_total


def get_data(thrust, bound, episodes):
    """Gets experiment data and returns dictionary of polished data."""
    tic = time.time()
    with Pool(16) as p:
        all_data = p.map(one_episode, [(thrust, bound)]*episodes)
    toc = time.time()
    rendez_counter = 0
    rendez_delta_v_total = 0
    for episode_data in all_data:
        if episode_data[0] > 0:
            rendez_counter += 1
            rendez_delta_v_total += episode_data[1]
    print('Time: ', toc-tic, ' Bound: ', bound, ' Thrust: ', thrust)
    return {'episodes': episodes, 'tot_time': toc-tic,
            'bound': bound, 'thrust': thrust,
            'num_rendez': rendez_counter,
            'total_rendez_delta_v': rendez_delta_v_total}  #TODO change data!! initially had total_return_delta_v


def main_exp():
    """"Main experiment function."""
    data_dict = {}
    episodes = int(1e5)
    for thrust in [5, 7.5, 10, 15, 20]:
        for bound in [1e5, 5e5, 1e6, 5e6, 1e7]:
            data = get_data(thrust, bound, episodes)
            data_dict[(thrust, bound)] = data
    with open('exp_2_1_data.pkl', 'wb') as f:
        pickle.dump(data_dict, f)


if __name__ == "__main__":
    main_exp()
