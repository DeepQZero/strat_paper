from multiprocessing import Pool
import time
import pickle

import numpy as np

from lib import dynamics as dyn
from experiments.section_2.exp_2_1_rand.exp_2_1_env import BaseSpaceEnv


def one_episode(params):
    """Gets data from one episode."""
    thrust, zones = params
    env = BaseSpaceEnv()
    # rand_start_ang = 0
    ran = np.random.randint(8)
    rand_start_ang = np.pi/8 * 8
    # rand_start_ang = [-np.pi/4, 0, np.pi/4][ran]
    # rand_start_ang = np.random.uniform(-np.pi/4, np.pi/4)  # TODO Change back!
    temp_mobile = [-dyn.GEO, 0.0, 0.0, -dyn.BASE_VEL_Y]
    px, py = dyn.rotate(temp_mobile[0], temp_mobile[1], rand_start_ang)
    vx, vy = dyn.rotate(temp_mobile[2], temp_mobile[3], rand_start_ang)
    mobile = np.array([px, py, vx, vy])
    base = np.array([dyn.GEO, 0.0, 0.0, dyn.BASE_VEL_Y])
    time_step = 0
    _ = env.det_reset(mobile, base, time_step)
    done = False
    fuel_total = 0
    zone_dict = {angle: (0, 0, False) for angle in zones}
    while not done:
        rand_thrust = np.random.uniform(0, thrust)
        rand_angle = np.random.uniform(0, 2 * np.pi)
        rand_act = np.array([np.cos(rand_angle), np.sin(rand_angle)]) * rand_thrust
        fuel_total += dyn.vec_norm(rand_act)
        state, reward, done, info = env.step(rand_act)
        zone = dyn.abs_angle_diff(state[0:2], state[4:6])
        for zone_key in zone_dict:
            if not zone_dict[zone_key][2]:
                if (zone_key - np.pi/8 < zone < zone_key) and \
                        abs(dyn.vec_norm(state[0:2]) - dyn.GEO) < 5e6:
                    zone_dict[zone_key] = (state[8], fuel_total, True)
    return zone_dict


def get_data(thrust, zones, episodes):
    """Gets experiment data and returns dictionary of polished data."""
    # get experiment raw data
    tic = time.time()
    with Pool(16) as p:
        all_data = p.map(one_episode, [(thrust, zones)]*episodes)
    toc = time.time()

    # total episode data
    zone_totals = {zone_key: {'first_tot': 0, 'fuel_tot': 0, 'count': 0}
                   for zone_key in zones}
    for episode_data in all_data:
        for zone_key in zones:
            zone_totals[zone_key]['first_tot'] += episode_data[zone_key][0]
            zone_totals[zone_key]['fuel_tot'] += episode_data[zone_key][1]
            if episode_data[zone_key][2]:
                zone_totals[zone_key]['count'] += 1

    # compute statistics
    zone_stats = {}
    for zone_key in zones:
        avg_first = 0 if zone_totals[zone_key]['count'] == 0 else \
            zone_totals[zone_key]['first_tot'] / zone_totals[zone_key]['count']
        avg_thrust = 0 if zone_totals[zone_key]['count'] == 0 else \
            zone_totals[zone_key]['fuel_tot'] / zone_totals[zone_key]['count']
        num_entrances = zone_totals[zone_key]['count']
        zone_stats[zone_key] = {'avg_first': avg_first, 'avg_thrust': avg_thrust, 'num_entrances': num_entrances}
    print('Time: ', toc-tic, ' Thrust: ', thrust)
    return zone_stats


def main_exp():
    """"Main experiment function."""
    data_dict = {}
    episodes = int(1e4)
    zones = [round(np.pi/8 * i, 2) for i in range(1, 8)]
    for thrust in [5, 7.5, 10, 15, 20]:
        data = get_data(thrust, zones, episodes)
        data_dict[thrust] = data
    with open('exp_3_2_data.pkl', 'wb') as f:
        pickle.dump(data_dict, f)


if __name__ == "__main__":
    main_exp()