from multiprocessing import Pool
import time

import numpy as np

from lib import dynamics as dyn
from experiment_1_env import Env


def one_run(params):
    bound, upper_thrust, lower_thrust, angle_diffs = params
    env = Env()
    _ = env.reset()
    done = False
    fuel_total = 0
    flag_dict = {angle: (0, 0, False) for angle in angle_diffs}
    while not done:
        action = np.random.uniform(lower_thrust, upper_thrust, 2)
        fuel_total += dyn.norm(action[0], action[1])
        state, reward, done, info = env.step(action)
        angle = dyn.angle_diff(state[0], state[1], state[4], state[5])
        for key in flag_dict:
            if not flag_dict[key][2]:  # not reached desired angle
                if angle < key:
                    flag_dict[key] = (state[12], fuel_total, True)
    return flag_dict


def main(bound, upper_thrust, lower_thrust, angle_diffs, times):
    tic = time.time()
    with Pool(16) as p:
        xs = p.map(one_run, [(bound, upper_thrust, lower_thrust, angle_diffs)]*times)  # list of dictionaries
        # print(xs)
    toc = time.time()
    sample_dict = xs[0]
    totals = {key: [0, 0] for key in sample_dict}
    for d in xs:
        for key in d:
            totals[key][0] += d[key][0]
            totals[key][1] += d[key][1]
    for key in totals:
        print(key, totals[key][0] / times, totals[key][1] / times)
    # print('Time: ', toc-tic, ' Bound: ', bound, ' Upper Thrust: ', upper_thrust, ' Lower Thrust: ', lower_thrust,
    #       ' Passive Prop: ', passive_prop,
    #       ' Captures: ', tot_1, ' Returns: ', tot_2, ' Capture Delta V: ', tot_3, ' Return Delta V: ', tot_4)


def main2():
    TIMES = 1000
    #for UPPER_THRUST in [10, 5, 2, np.sqrt(2), 1]:
    for UPPER_THRUST in [2]:
        ANGLE_DIFFS = [round(np.pi/8 * i, 2) for i in range(1, 8)]
        LOWER_THRUST = 0
        BOUND = 1e5
        main(BOUND, UPPER_THRUST, LOWER_THRUST, ANGLE_DIFFS, TIMES)


if __name__ == "__main__":
    main2()
