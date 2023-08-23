import pickle

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.spaces import Discrete

from lib import dynamics

GEO = 42.164e6
BASE_VEL_Y = 3.0746e3
MU = 3.9860e14


class Env(gym.Env):
    def __init__(self,
                 step_length=3600,
                 discretization=60,
                 max_turns=24*28,
                 give_capture_reward=False):
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(1, 14), dtype=np.float32)
        self.action_space = Discrete(9)
        self.CAP_RAD = 1e5
        self.DISCRETIZATION = discretization
        self.UPDATE_LENGTH = step_length / discretization
        self.MAX_TURNS = max_turns
        self.caught = None
        self.current_turn = None
        self.unit = None
        self.enemy_base = None
        self.friendly_base = None
        self.fuel = None
        self.give_capture_reward = give_capture_reward
        self.action_mag_hist = []
        self.result_hist = []
        self.num_timesteps = 0
        self.PICKLE_NAME = "exp_2_4_data.pkl"

    def reset(self, state=np.array([-GEO, 0.0, 0.0, -BASE_VEL_Y,
                                   -GEO, 0.0, 0.0, -BASE_VEL_Y,
                                   GEO, 0.0, 0.0, BASE_VEL_Y,
                                   0, 0]), seed=None, options=None):
        """Resets environment. Returns first observation per Gym Standard."""
        # TODO: local vs global variables - what is happening???
        self.unit = np.array([-GEO, 0.0, 0.0, -BASE_VEL_Y])
        self.friendly_base = np.array([-GEO, 0.0, 0.0, -BASE_VEL_Y])
        self.enemy_base = np.array([GEO, 0.0, 0.0, BASE_VEL_Y])
        #self.unit = state[0:4]
        #self.friendly_base = state[4:8]
        #self.enemy_base = state[8:12]
        self.caught = int(state[12])
        self.current_turn = 0 # TODO CHANGE BACK!!!
        self.fuel = 10000
        return self.det_obs(), None

    def det_obs(self) -> np.ndarray:
        """Returns observation by Gym standard."""
        angle = np.arctan2(self.enemy_base[1], self.enemy_base[0])
        unit = self.unit_obs(self.unit, angle)
        friendly_base = self.unit_obs(self.friendly_base, angle)
        enemy_base = self.unit_obs(self.enemy_base, angle)
        return np.concatenate((unit, friendly_base, enemy_base,
                               [self.caught], [self.current_turn]))

    def unit_obs(self, unit, angle):
        return unit
        # unit_new_x_y = rotate(*unit[0:2], -angle)
        # unit = [*unit_new_x_y, *unit[2:]]
        # return unit

    def step(self, action):
        self.num_timesteps += 1
        self.action_mag_hist.append(self.score_action(action))
        if (self.num_timesteps % int(5e3)) == 0:
            self.result_hist.append([self.num_timesteps, np.average(self.action_mag_hist)])
            self.action_mag_hist = []
            print(self.result_hist)
        if self.num_timesteps == int(1e5):
            pickle.dump(self.result_hist, open(self.PICKLE_NAME, "wb"))
        rotated_thrust = self.decode_action(action)
        self.unit[2:4] += rotated_thrust
        self.unit = self.prop_unit(self.unit)
        self.friendly_base = self.prop_unit(self.friendly_base)
        self.enemy_base = self.prop_unit(self.enemy_base)
        self.current_turn += 1
        neg_fuel = self.score_action(action)
        self.fuel -= self.score_action(action)
        if dynamics.distance(self.unit[0:2], self.enemy_base[0:2]) < self.CAP_RAD and self.caught == 0:
            self.caught = 1
            print("CAPTURE")
        elif self.caught == 1 and dynamics.distance(self.unit[0:2], self.friendly_base[0:2]) < self.CAP_RAD:
            self.caught = 2
            print("VICTORY!!!!!", self.current_turn)
        return self.det_obs(), self.det_reward(action), self.is_done(), False, {}

    def is_done(self):
        return self.current_turn == self.MAX_TURNS #or self.caught == 2 or self.fuel <= 0

    def det_reward(self, action):
        if self.give_capture_reward and self.caught == 2:
            return 10 - self.score_action(action)
        else:
            return -1 * self.score_action(action)

    def prop_unit(self, unit):
        return dynamics.propagate(unit[0:4], self.DISCRETIZATION, self.UPDATE_LENGTH)

    def score_action(self, act):
        new_act = self.decode_action(act)
        return np.sqrt(new_act[0]**2) + np.sqrt(new_act[1]**2)

    def decode_action(self, act):
        if act == 0:
            action = [-1.0, -1.0]
        elif act == 1:
            action = [-1.0, 0.0]
        elif act == 2:
            action = [-1.0, 1.0]
        elif act == 3:
            action = [0.0, -1.0]
        elif act == 4:
            action = [0.0, 0.0]
        elif act == 5:
            action = [0.0, 1.0]
        elif act == 6:
            action = [1.0, -1.0]
        elif act == 7:
            action = [1.0, 0.0]
        else:
            action = [1.0, 1.0]
        action[0] *= 1
        action[1] *= 1
        angle = np.arctan2(self.unit[3], self.unit[2])
        action = dynamics.rotate(*action, angle)
        return action


    @staticmethod
    def angle_diff(state):
        x = np.arctan2(state[1], state[0])
        y = np.arctan2(state[9], state[8])
        abs_diff = np.abs(x - y)
        # print(x, y, abs_diff)
        return min((2 * np.pi) - abs_diff, abs_diff)

    @staticmethod
    def angle_diff1(state):
        x = np.arctan2(state[1], state[0])
        y = np.arctan2(state[5], state[4])
        abs_diff = np.abs(x - y)
        # print(x, y, abs_diff)
        return min((2 * np.pi) - abs_diff, abs_diff)