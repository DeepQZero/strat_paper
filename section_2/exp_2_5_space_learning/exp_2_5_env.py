import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from exp_2_4_collapse import exp_2_4_env


class Env(exp_2_4_env.Env):
    def __init__(self):
        super().__init__(give_capture_reward=True)
        self.PICKLE_NAME = "exp_2_5_data.pkl"
