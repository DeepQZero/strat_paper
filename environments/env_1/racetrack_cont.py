import numpy as np
from numba import njit
import pickle
from lib import dynamics
import gym

SCALING = 6
THRUST_FACTOR = 10

class Env(gym.Env):
    GEO = dynamics.GEO
    BASE_VEL_Y = dynamics.BASE_VEL_Y

    def __init__(self,
                 step_length=3600*SCALING,
                 max_turns=28*24/SCALING,
                 discretization=10*SCALING):
        self.CAP_RAD = 1e5
        self.DISCRETIZATION = discretization
        self.UPDATE_LENGTH = step_length / discretization
        self.MAX_TURNS = max_turns
        self.caught = None
        self.current_turn = None
        self.unit = None
        self.enemy_base = None
        self.friendly_base = None
        self.action_space = gym.spaces.Box(-np.pi, np.pi, shape=(2,))
        self.observation_space = gym.spaces.Box(-1e8, 1e8, shape=(10,))

    def reset(self) -> np.ndarray:
        """Resets environment. Returns first observation per Gym Standard."""
        self.unit = np.array([dynamics.GEO, 0.0, 0.0,dynamics.BASE_VEL_Y])
        self.friendly_base = np.array([-dynamics.GEO, 0.0, 0.0,-dynamics.BASE_VEL_Y])
        self.enemy_base = np.array([dynamics.GEO, 0.0, 0.0,dynamics.BASE_VEL_Y])
        self.caught = int(1)
        self.current_turn = int(0)
        return self.det_obs()

    def step(self, action) -> tuple:
        thrust = self.decode_action(action)
        self.unit[2:4] += thrust
        self.unit = self.prop_unit(self.unit)
        self.friendly_base = self.prop_unit(self.friendly_base)
        self.enemy_base = self.prop_unit(self.enemy_base)
        self.current_turn += 1
        if dynamics.distance(self.unit[0:2], self.enemy_base[0:2]) < self.CAP_RAD and self.caught == 0:
            self.caught = 1
            # print("CAPTURE", self.current_turn)
        elif self.caught == 1 and dynamics.distance(self.unit[0:2], self.friendly_base[0:2]) < self.CAP_RAD:
            self.caught = 2
            print("VICTORY!!!!!", self.current_turn)
        return self.det_obs(), self.det_reward(), self.is_done(), {}

    def is_out_bounds(self):
        radius = np.linalg.norm(self.unit[0:2])
        return (radius < dynamics.GEO - dynamics.GEO_BOUND) or (radius > dynamics.GEO + dynamics.GEO_BOUND)

    def det_obs(self) -> np.ndarray:
        return np.concatenate((self.unit[0:4], self.friendly_base[0:4], [self.caught], [self.current_turn]))
    #
    # def norm_unit(self, unit):
    #     return np.concatenate((unit[0:2]/dynamics.GEO, unit[2:4]/dynamics.BASE_VEL_Y))
    #
    # def det_obs_1(self) -> np.ndarray:
    #     """Returns observation by Gym standard."""
    #     angle = np.arctan2(self.enemy_base[1], self.enemy_base[0])
    #     unit = self.unit_obs(self.unit, angle)
    #     friendly_base = self.unit_obs(self.friendly_base, angle)
    #     enemy_base = self.unit_obs(self.enemy_base, angle)
    #     return np.concatenate((unit, friendly_base, enemy_base,
    #                            [self.caught], [self.current_turn]))

    def unit_obs(self, unit, angle):  # TODO: adjust to rotate goal at each step
        unit_new_x_y = dynamics.rotate(*unit[0:2], -angle)
        unit = [*unit_new_x_y, *unit[2:]]
        return unit

    def is_done(self):
        return self.caught == 2 or self.is_out_bounds() or self.current_turn >= self.MAX_TURNS

    def det_reward(self):
        if self.caught == 2:
            return 1
        elif self.is_done():
            return 0
        else:
            return 0

    def prop_unit(self, unit):
        return dynamics.propagate(unit[0:4], self.DISCRETIZATION, self.UPDATE_LENGTH)

    def score_action(self, act):
        new_act = self.decode_action(act)
        return np.sqrt(new_act[0]**2) + np.sqrt(new_act[1]**2)

    def decode_action(self, act):
        return [act[0]/np.pi * THRUST_FACTOR * np.cos(act[1]),
                act[0]/np.pi * THRUST_FACTOR * np.sin(act[1])]


def test():
    env = Env()
    for i in range(100000):
        if (i % 100) == 0:
            print(i)
        start_state = env.reset()
        done = False
        steps = 0
        while not done:
            steps += 1
            action = np.random.uniform(-np.pi, np.pi, 2)
            state, reward, done, info = env.step(action)
            # print(state, np.linalg.norm(state[0:2]))
        # if steps == 1:
        #     print(state)


if __name__ == "__main__":
    test()

