from multiprocessing import Pool
import time
import pickle

import numpy as np

from lib import dynamics as dyn
from experiments.section_2.exp_2_2_pass.exp_2_2_env import Env


BOUND = 1e6
LOWER_THRUST = 0
UPPER_THRUST = 1


def one_run(params):
    bound, upper_thrust, lower_thrust, passive_prop = params
    env = Env()
    _ = env.reset()
    done = False
    is_halfway = False
    capture_flag = 0
    return_flag = 0
    fuel_total = 0
    while not done:
        if np.random.uniform(0, 1) < passive_prop:
            action = np.array([0, 0])
        else:
            action = np.random.uniform(lower_thrust, upper_thrust, 2)
        fuel_total += dyn.norm(action[0], action[1])
        state, reward, done, info = env.step(action)
        if not is_halfway and (dyn.norm(state[0], state[1], state[4], state[5]) <= bound):
            is_halfway = True
            capture_flag = state[12]  # current turn
        if is_halfway and (dyn.norm(state[0], state[1], state[8], state[9]) <= bound):
            done = True
            return_flag = state[12]
    return (capture_flag, return_flag, fuel_total)


def main(bound, upper_thrust, lower_thrust, passive_prop, times):
    tic = time.time()
    with Pool(16) as p:
        xs = p.map(one_run, [(bound, upper_thrust, lower_thrust, passive_prop)]*times)
        # print(xs)
    toc = time.time()
    tot_1 = 0
    tot_2 = 0
    tot_3 = 0
    tot_4 = 0
    for x in xs:
        if x[0] > 0:
            tot_1 += 1
            tot_3 += x[2]
        if x[1] > 0:
            tot_2 += 1
            tot_4 += x[2]
    print('Time: ', toc-tic, ' Bound: ', bound, ' Upper Thrust: ', upper_thrust, ' Lower Thrust: ', lower_thrust,
          ' Passive Prop: ', passive_prop,
          ' Captures: ', tot_1, ' Returns: ', tot_2, ' Capture Delta V: ', tot_3, ' Return Delta V: ', tot_4)
    return {'times': times, 'tot_time': toc-tic, 'bound': bound, 'upper_thrust': upper_thrust, 'captures': tot_1,
            'returns': tot_2, 'capture_delta_v': tot_3, 'return_delta_v': tot_4, 'passive_prop': passive_prop}


def main2():
    pickle_dict = {}
    TIMES = 100
    for UPPER_THRUST in [10, 5, 2, np.sqrt(2), 1]:
        for PASSIVE_PROP in [0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]:
            LOWER_THRUST = 0
            BOUND = 1e5
            one_data = main(BOUND, UPPER_THRUST, LOWER_THRUST, PASSIVE_PROP, TIMES)
            pickle_dict[(PASSIVE_PROP, UPPER_THRUST)] = one_data
    with open('exp_2_2_data.pkl', 'wb') as f:
        pickle.dump(pickle_dict, f)


if __name__ == "__main__":
    main2()
