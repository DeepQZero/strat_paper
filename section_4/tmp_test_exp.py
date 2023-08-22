from multiprocessing import Pool
import time
import pickle

import numpy as np

from lib import dynamics as dyn
from tmp_env import Env


def one_episode(i):
    """Gets data from one episode."""
    MAX_FUEL = 200
    MAX_TURNS = 112
    thrust = 10
    np.random.seed()
    env = Env()
    state = env.reset()
    done = False
    while not done:
        rand_thrust = np.random.uniform(0, thrust)
        rand_angle = np.random.uniform(0, 2 * np.pi)
        rand_act = np.array([np.cos(rand_angle), np.sin(rand_angle)]) * rand_thrust
        # if np.random.uniform(0, 1) < 0.75:
        #     rand_act = np.array([0.0, 0.0])
        if ((MAX_FUEL - state[13]) / MAX_FUEL) < 0.5 and (dyn.abs_angle_diff(state[0:2], state[8:10]) > (np.pi/2)):
            rand_act = np.array([0.0, 0.0])
        if ((MAX_FUEL - state[13]) / MAX_FUEL) < 0.25 and (dyn.abs_angle_diff(state[0:2], state[8:10]) > (np.pi/4)):
            rand_act = np.array([0.0, 0.0])
        if ((MAX_FUEL - state[13]) / MAX_FUEL) < 0.125 and (dyn.abs_angle_diff(state[0:2], state[8:10]) > (np.pi/8)):
            rand_act = np.array([0.0, 0.0])
        state, reward, done, info = env.step(rand_act)

        angle = dyn.abs_angle_diff(state[0:2], state[8:10])
        dist = dyn.vec_norm(state[0:2])
        in_zone = abs(dist - dyn.GEO) < 5e6
        angle_left_proportion = angle / np.pi
        fuel_left_proportion = (MAX_FUEL*1 - state[13]) / (MAX_FUEL*1)
        turn_left_proportion = (MAX_TURNS - state[12]) / MAX_TURNS
        good_fuel = angle_left_proportion < fuel_left_proportion
        good_turn = angle_left_proportion < turn_left_proportion
        if in_zone and good_fuel and good_turn and abs(dyn.vec_norm(state[0:2]) - dyn.GEO) < 1e7:
            print(angle_left_proportion, "{:e}".format(abs(dyn.vec_norm(state[0:2]) - dyn.GEO)),
                  "{:e}".format(dyn.vec_norm(state[0:2] - state[8:10])), fuel_left_proportion, turn_left_proportion)
            print('SUCCESS', i)
            # done = True
        if dyn.vec_norm(state[0:2] - state[8:10]) < 5e5:
            print('CAPTURE ', i, fuel_left_proportion)
            done = True
    return None


def main_exp():
    """Gets experiment data and returns dictionary of polished data."""
    # get experiment raw data
    tic = time.time()
    with Pool(4) as p:
        _ = p.map(one_episode, list(range(int(1e6))))
    toc = time.time()
    print(tic-toc)


if __name__ == "__main__":
    main_exp()
