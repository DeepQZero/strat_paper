import copy

import numpy as np
import gymnasium as gym

from lib import dynamics as dyn


class BaseSpaceEnv(gym.Env):
    def __init__(self, step_len: int = 10800, dis: int = 180,
                 max_turns: int = 4*28, give_capture_reward=False) -> None:
        self.DIS = dis
        self.UP_LEN = step_len / dis
        self.MAX_TURNS = max_turns
        self.mobile = None
        self.base = None
        self.time_step = None
        self.give_capture_reward = give_capture_reward # TODO IMPLEMENT
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 14), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(9)

    def reset(self, seed=None, options=None) -> np.ndarray:
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
        return self.det_obs(), self.det_reward(), self.is_done(), False, {}

    def is_done(self) -> bool:
        """Determines if episode has reached termination."""
        return self.time_step == self.MAX_TURNS

    def det_reward(self, action) -> float:
        """Returns reward at current time step."""
        return -1 * self.score_action(action)

    def score_action(self, act):
        new_act = self.decode_action(act)
        return np.sqrt(new_act[0]**2) + np.sqrt(new_act[1]**2)

    def prop_unit(self, unit: np.ndarray) -> np.ndarray:
        """Propagates a given unit state forward one time step."""
        return dyn.propagate(unit[0:4], self.DIS, self.UP_LEN)
