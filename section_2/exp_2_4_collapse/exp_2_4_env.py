import os
import sys
import pickle

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np

from exp_2_space_env import Env as BaseEnv

GEO = 42.164e6
BASE_VEL_Y = 3.0746e3
MU = 3.9860e14


class SparseEnv(BaseEnv):
    def __init__(self, step_len=10800, dis=180, max_turns=8*14, log_increment=5e2, log_time=2e5,
                 give_capture_reward=True, dense_reward=False, do_drifting=False, fuel_penalty=False):
        super().__init__(step_len=step_len, dis=dis, max_turns=max_turns, capture_reward=True,
                         drifting=do_drifting, dense_reward=False, add_fuel_penalty=fuel_penalty)
        self.CAPTURE_RADIUS = 1e5
        self.SIGMA = 0.01
        self.action_mag_hist = []
        self.result_hist = []
        self.num_timesteps = 0
        self.PICKLE_NAME = "exp_2_4_data.pkl"
        self.LOG_INCREMENT = log_increment
        self.LOG_TIME = log_time

    def step(self, action):
        self.total_fuel = 0.0
        self.num_timesteps += 1
        self.action_mag_hist.append(np.abs(action[0]))
        if (self.num_timesteps % int(self.LOG_INCREMENT)) == 0:
            self.result_hist.append([self.num_timesteps, (self.action_mag_hist)])  # np.average(self.action_mag_hist)
            print(np.average(self.action_mag_hist))
            self.action_mag_hist = []
        if self.num_timesteps == int(self.LOG_TIME):
            pickle.dump(self.result_hist, open(self.PICKLE_NAME, "wb"))
        return super().step(action)

