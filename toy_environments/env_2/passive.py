import numpy as np
from numba import njit
import pickle
from lib import dynamics
import gym


class Env(gym.Env):
    GEO = dynamics.GEO
    BASE_VEL_Y = dynamics.BASE_VEL_Y

    def __init__(self,
                 step_length=3600,
                 max_turns=24*28,
                 discretization=60):
        self.CAP_RAD = 1e5
        self.DISCRETIZATION = discretization
        self.UPDATE_LENGTH = step_length / discretization
        self.MAX_TURNS = max_turns
        self.caught = None
        self.current_turn = None
        self.unit = None
        self.enemy_base = None
        self.friendly_base = None
        self.action_space = gym.spaces.Discrete(9)
        self.observation_space = gym.spaces.Box(-1000, 1000, shape=(8,))

    '''def reset(self, state=np.array([-GEO, 0.0, 0.0, -BASE_VEL_Y,
                                   -GEO, 0.0, 0.0, -BASE_VEL_Y,
                                   GEO, 0.0, 0.0, BASE_VEL_Y,
                                   0, 0])) -> np.ndarray:
        """Resets environment. Returns first observation per Gym Standard."""
        self.unit = state[0:4]
        self.friendly_base = state[4:8]
        self.enemy_base = state[8:12]
        self.caught = int(state[12])
        self.current_turn = int(state[13])
        return self.det_obs()'''

    def reset(self, state = None) -> np.ndarray:
        GEO = dynamics.GEO
        BASE_VEL_Y = dynamics.BASE_VEL_Y
        """Resets environment. Returns first observation per Gym Standard."""
        self.unit = np.array([-GEO, 0.0, 0.0, -BASE_VEL_Y])
        self.friendly_base = np.array([-GEO, 0.0, 0.0, -BASE_VEL_Y])
        self.enemy_base = np.array([GEO, 0.0, 0.0, BASE_VEL_Y])
        self.caught = 0
        self.current_turn = 0
        return np.concatenate((self.norm_unit(self.unit), self.norm_unit(self.friendly_base), self.norm_unit(self.enemy_base),
                        [self.caught], [self.current_turn]))

    def step(self, action):
        rotated_thrust = self.decode_action(action)
        self.unit[2:4] += rotated_thrust
        self.unit = self.prop_unit(self.unit)
        self.friendly_base = self.prop_unit(self.friendly_base)
        self.enemy_base = self.prop_unit(self.enemy_base)
        self.current_turn += 1
        if dynamics.distance(self.unit[0:2], self.enemy_base[0:2]) < self.CAP_RAD and self.caught == 0:
            self.caught = 1
            print("CAPTURE", self.current_turn)
        elif self.caught == 1 and dynamics.distance(self.unit[0:2], self.friendly_base[0:2]) < self.CAP_RAD:
            self.caught = 2
            print("VICTORY!!!!!", self.current_turn)
        return self.det_obs(), self.det_reward(), self.is_done(), {}

    def is_out_bounds(self):
        radius = np.linalg.norm(self.unit[0:2])
        return (radius < dynamics.GEO - dynamics.GEO_BOUND) or (radius > dynamics.GEO + dynamics.GEO_BOUND)

    def det_obs(self) -> np.ndarray:
        return np.concatenate((self.norm_unit(self.unit), self.norm_unit(self.friendly_base), self.norm_unit(self.enemy_base),
                               [self.caught], [self.current_turn]))

    def norm_unit(self, unit):
        return np.concatenate((unit[0:2]/dynamics.GEO, unit[2:4]/dynamics.BASE_VEL_Y))

    def det_obs_1(self) -> np.ndarray:
        """Returns observation by Gym standard."""
        angle = np.arctan2(self.enemy_base[1], self.enemy_base[0])
        unit = self.unit_obs(self.unit, angle)
        friendly_base = self.unit_obs(self.friendly_base, angle)
        enemy_base = self.unit_obs(self.enemy_base, angle)
        return np.concatenate((unit, friendly_base, enemy_base,
                               [self.caught], [self.current_turn]))

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
        action[0] *= 10
        action[1] *= 10
        angle = np.arctan2(self.unit[3], self.unit[2])
        action = dynamics.rotate(*action, angle)
        return action


def angle_diff(state):
    x = np.arctan2(state[1], state[0])
    y = np.arctan2(state[9], state[8])
    abs_diff = np.abs(x - y)
    # print(x, y, abs_diff)
    return min((2 * np.pi) - abs_diff, abs_diff)


def main():
    success_counter = 0
    env = Env()
    for i in range(100):
        print(i)
        state = env.reset()
        done = False
        j=0
        while not done:
            j+= 1
            if np.random.uniform(0, 1) < 0.95:
                action = 4
            else:
                action = np.random.randint(9)
            next_state, reward, done, info, = env.step(action)
            # print(next_state)
            # print(angle_diff(next_state))
            state = next_state
            if angle_diff(next_state) < 0.1:
                print(i, 'WIN', j)
                done = True
    return success_counter


if __name__ == "__main__":
    num_success = main()