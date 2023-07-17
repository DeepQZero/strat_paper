from multiprocessing import Pool
import time
import pickle

import numpy as np

from lib import dynamics as dyn
from experiments.section_2.exp_2_1_rand.exp_2_1_env import Env


BOUND = 1e6
LOWER_THRUST = 0
UPPER_THRUST = 1


def one_run(params):
    bound, upper_thrust, lower_thrust = params
    env = Env()
    _ = env.reset()
    done = False
    is_halfway = False
    capture_flag = 0
    return_flag = 0
    fuel_total = 0
    while not done:
        random_act = np.random.uniform(lower_thrust, upper_thrust, 2)
        fuel_total += dyn.norm(random_act[0], random_act[1])
        state, reward, done, info = env.step(random_act)
        if not is_halfway and (dyn.norm(state[0], state[1], state[4], state[5]) <= bound):
            is_halfway = True
            capture_flag = state[12]  # current turn
        if is_halfway and (dyn.norm(state[0], state[1], state[8], state[9]) <= bound):
            done = True
            return_flag = state[12]
    return (capture_flag, return_flag, fuel_total)


def main(bound, upper_thrust, lower_thrust, times):
    tic = time.time()
    with Pool(16) as p:
        xs = p.map(one_run, [(bound, upper_thrust, lower_thrust)]*times)
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
          ' Captures: ', tot_1, ' Returns: ', tot_2, ' Capture Delta V: ', tot_3, ' Return Delta V: ', tot_4)
    return {'times': times, 'tot_time': toc-tic, 'bound': bound, 'upper_thrust': upper_thrust, 'captures': tot_1,
            'returns': tot_2, 'capture_delta_v': tot_3, 'return_delta_v': tot_4}


def main2():
    pickle_dict = {}
    TIMES = 100
    for UPPER_THRUST in [10, 5, 2, np.sqrt(2), 1]:
        for BOUND in [1e7, 1e6, 1e5, 1e4, 1e3]:
            LOWER_THRUST = 0
            one_data = main(BOUND, UPPER_THRUST, LOWER_THRUST, TIMES)
            pickle_dict[(BOUND, UPPER_THRUST)] = one_data
    with open('exp_2_1_data.pkl', 'wb') as f:
        pickle.dump(pickle_dict, f)


if __name__ == "__main__":
    main2()
