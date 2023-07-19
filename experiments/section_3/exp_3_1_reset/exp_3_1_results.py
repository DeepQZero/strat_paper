from multiprocessing import Pool
import time
import pickle

import numpy as np

from lib import dynamics as dyn
from experiments.section_3.exp_3_1_reset.exp_3_1_env import Env


def one_run(params):
    bound, upper_thrust, lower_thrust, angle_diffs, random = params
    env = Env()
    if random:
        _ = env.reset()
    else:
        # rand_choice = np.random.randint(2)
        # rand_ang = [np.pi/2, 0][rand_choice]
        rand_ang = np.random.uniform(-np.pi/4, np.pi/4)
        turn = 0
        px, py = dyn.rotate(-dyn.GEO, 0, rand_ang)
        vx, vy = dyn.rotate(0, -dyn.BASE_VEL_Y, rand_ang)
        unit = np.array([px, py, vx, vy])
        base = np.array([dyn.GEO, 0.0, 0.0, dyn.BASE_VEL_Y])
        return_base = np.array([-dyn.GEO, 0.0, 0.0, -dyn.BASE_VEL_Y])
        _ = env.det_reset(turn, unit, base, return_base)
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
                if key - np.pi/8 < angle < key:
                    flag_dict[key] = (state[12], fuel_total, True)
    return flag_dict


def main(bound, upper_thrust, lower_thrust, angle_diffs, times, random):
    tic = time.time()
    with Pool(16) as p:
        xs = p.map(one_run, [(bound, upper_thrust, lower_thrust, angle_diffs, random)]*times)  # list of dictionaries
        # print(xs)
    toc = time.time()
    sample_dict = xs[0]
    totals = {key: [0, 0, 0] for key in sample_dict}
    for d in xs:
        for key in d:
            totals[key][0] += d[key][0]  # total turns it was in the zone
            totals[key][1] += d[key][1]  # total delta v it was in the zone
            if d[key][0] > 0:
                totals[key][2] += 1   # times it was in the zone
    stats_dict = {}
    for key in totals:
        first = 0 if totals[key][2] == 0 else totals[key][0] / totals[key][2]
        second = 0 if totals[key][2] == 0 else totals[key][1]/ totals[key][2]
        third = totals[key][2]
        small_dict = {'first': first, 'second': second, 'third': third}
        stats_dict[key] = small_dict
    return stats_dict


def main2():
    pickle_dict = {}
    TIMES = 1000
    for UPPER_THRUST in [10, 5, 2, np.sqrt(2), 1]:
        ANGLE_DIFFS = [round(np.pi/8 * i, 2) for i in range(1, 8)]
        LOWER_THRUST = 0
        BOUND = 1e5
        one_data = main(BOUND, UPPER_THRUST, LOWER_THRUST, ANGLE_DIFFS, TIMES, True)
        pickle_dict[UPPER_THRUST] = one_data
    with open('exp_3_1_data.pkl', 'wb') as f:
        pickle.dump(pickle_dict, f)
    # two_data = main(BOUND, UPPER_THRUST, LOWER_THRUST, ANGLE_DIFFS, TIMES, False)
    # with open('exp_3_1_2_data.pkl', 'wb') as f:
    #     pickle.dump(pickle_dict, f)


if __name__ == "__main__":
    main2()
