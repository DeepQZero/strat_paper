from multiprocessing import Pool
import time
import pickle

import numpy as np

from lib import dynamics as dyn
from experiments.section_3.exp_3_3_cluster.exp_3_3_env import Env


def one_episode(thrust):
    """Gets data from one episode."""
    MAX_FUEL = 100
    MAX_TURNS = 112
    env = Env()
    _ = env.reset()
    done = False
    while not done:
        rand_thrust = np.random.uniform(0, thrust)
        rand_angle = np.random.uniform(0, 2 * np.pi)
        rand_act = np.array([np.cos(rand_angle), np.sin(rand_angle)]) * rand_thrust
        state, reward, done, info = env.step(rand_act)

        angle = dyn.abs_angle_diff(state[0:2], state[8:10])
        dist = dyn.vec_norm(state[0:2])
        in_zone = abs(dist - dyn.GEO) < 5e6
        angle_left_proportion = angle / np.pi
        fuel_left_proportion = (MAX_FUEL*1.25 - state[13]) / (MAX_FUEL*1.25)
        turn_left_proportion = (MAX_TURNS - state[12]) / MAX_TURNS
        good_fuel = angle_left_proportion < fuel_left_proportion
        good_turn = angle_left_proportion < turn_left_proportion
        # print(angle_left_proportion, fuel_left_proportion, turn_left_proportion)
        if in_zone and good_fuel and good_turn:
            print('SUCCESS')
    return None


def main_exp():
    """Gets experiment data and returns dictionary of polished data."""
    # get experiment raw data
    tic = time.time()
    thrust = 10
    with Pool(12) as p:
        _ = p.map(one_episode, [thrust]*1000)
    toc = time.time()
    print(tic-toc)

if __name__ == "__main__":
    main_exp()
