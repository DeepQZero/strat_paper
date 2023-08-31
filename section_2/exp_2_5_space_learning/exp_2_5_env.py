import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from exp_2_4_collapse import exp_2_4_env


class FuelPenEnv(exp_2_4_env.SparseEnv):
    def __init__(self):
        super().__init__(give_capture_reward=True, fuel_penalty=True)
        self.FUEL_MULTIPLIER = 10
        self.PICKLE_NAME = "exp_2_5_data.pkl"

    def det_reward(self, action):
        return self.det_fuel_rew(action)
