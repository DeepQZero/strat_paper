import os
import sys
import copy

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np

from exp_2_space_env import Env
from lib import dynamics as dyn


class BaseSpaceEnv(Env):
    def __init__(self, step_len: int = 10800, dis: int = 180,
                 max_turns: int = 4*28):
        super().__init__(step_len=step_len, dis=dis, max_turns=max_turns)

    def reset(self) -> np.ndarray:
        """Resets environment and returns observation per Gym Standard."""
        self.mobile = np.array([-dyn.GEO, 0.0, 0.0, -dyn.BASE_VEL_Y])
        self.base = np.array([dyn.GEO, 0.0, 0.0, dyn.BASE_VEL_Y])
        self.time_step = 0
        return self.det_obs()

    def det_reset(self, mobile, base, time_step) -> np.ndarray:
        """Resets environment deterministically."""
        self.mobile = copy.deepcopy(mobile)
        self.base = copy.deepcopy(base)
        self.time_step = copy.deepcopy(time_step)
        return self.det_obs()

    def det_obs(self) -> np.ndarray:
        """Returns observation per Gym standard."""
        return np.concatenate((self.mobile, self.base, [self.time_step]))

    def step(self, action: np.ndarray) -> tuple:
        """Advances environment forward one time step, returns Gym signals."""
        self.mobile[2:4] += action
        self.mobile = self.prop_unit(self.mobile)
        self.base = self.prop_unit(self.base)
        self.time_step += 1
        return self.det_obs(), self.det_reward(), self.is_done(), {}

    def is_done(self) -> bool:
        """Determines if episode has reached termination."""
        return self.time_step == self.MAX_TURNS

    def det_reward(self) -> float:
        """Returns reward at current time step."""
        if self.is_done():
            return 0.0
        else:
            return 0.0

