import os
import sys
import pickle

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from exp_2_space_env import Env as BaseEnv

GEO = 42.164e6
BASE_VEL_Y = 3.0746e3
MU = 3.9860e14


class Env(BaseEnv):
    def __init__(self, step_len=10800, dis=180, max_turns=8*14,
                 give_capture_reward=True, dense_reward=False, drifting=False, add_fuel_penalty=False):
        super().__init__(step_len=step_len, dis=dis, max_turns=max_turns, capture_reward=give_capture_reward,
                         drifting=drifting, dense_reward=dense_reward, add_fuel_penalty=add_fuel_penalty)
        self.CAPTURE_RADIUS = 1e5
        self.SIGMA = 0.00
        self.action_mag_hist = []
        self.result_hist = []
        self.num_timesteps = 0
        self.PICKLE_NAME = "exp_2_4_data.pkl"

    def step(self, action):
        self.num_timesteps += 1
        self.action_mag_hist.append(action)  # self.score_action(action)
        if (self.num_timesteps % int(2e3)) == 0:
            self.result_hist.append([self.num_timesteps, (self.action_mag_hist)])  # np.average(self.action_mag_hist)
            print(self.result_hist[-1])
            self.action_mag_hist = []
        if self.num_timesteps == int(1e5):
            pickle.dump(self.result_hist, open(self.PICKLE_NAME, "wb"))
        return super().step(action)

    def is_done(self):
        return self.time_step == self.MAX_TURNS or self.is_capture()

