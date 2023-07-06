import collections

import numpy as np
import sklearn

from lib import dynamics

GEO = 42.164e6
BASE_VEL_Y = 3.0746e3
MU = 3.9860e14

class Env:
    def __init__(self,
                 step_length=21600,
                 discretization=60,
                 max_turns=4*28):
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

    def reset(self, state=np.array([-GEO, 0.0, 0.0, -BASE_VEL_Y,
                                   -GEO, 0.0, 0.0, -BASE_VEL_Y,
                                   GEO, 0.0, 0.0, BASE_VEL_Y,
                                   0, 0])) -> np.ndarray:
        """Resets environment. Returns first observation per Gym Standard."""
        self.unit = state[0:4]
        self.friendly_base = state[4:8]
        self.enemy_base = state[8:12]
        self.caught = int(state[12])
        self.current_turn = 0 # TODO CHANGE BACK!!!
        self.fuel = 10000
        return self.det_obs()

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
        return self.det_obs(), self.det_reward(), self.is_done(), {}

    def is_done(self):
        return self.current_turn == self.MAX_TURNS or self.caught == 2 or self.fuel <= 0

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
        action[0] *= 1
        action[1] *= 1
        angle = np.arctan2(self.unit[3], self.unit[2])
        action = dynamics.rotate(*action, angle)
        return action


def buffer_stats(seen_states):
    angle_dict = {i:0 for i in range(360)}
    for s in seen_states:
        angle = np.arctan2(s[1], s[0]) + np.pi
        angle *= 180 / np.pi
        angle = int(np.floor(angle))
        radius = np.sqrt(s[0]**2) + np.sqrt(s[1]**2)
        if abs(radius - GEO) <= radius_threshold:
            angle_dict[angle] += 1
    counter = 0
    max_angle_reached = 0
    for key in angle_dict.keys():
        if angle_dict[key] > 0:
            counter += 1
            # TODO: Adjust the max angle reached to be centered around 180
            max_angle_reached = 180 - max(max_angle_reached, abs(key - 180))
    print("Angle dict: ", angle_dict)
    print("Counter: ", counter)
    print("Max angle reached: ", max_angle_reached)


def sample_from_buffer(buffer):
    for j in range(10):
        ind = np.random.randint(0, len(buffer))
        state = buffer[ind]
        radius = np.sqrt(state[0] ** 2) + np.sqrt(state[1] ** 2)
        if abs(radius - GEO) <= radius_threshold and state[12] < max_turns:
            return (True, state)
    return (False, None)


if __name__ == "__main__":
    env = Env()
    max_capacity = 10000
    reset_prob = 0.10
    radius_threshold = 10e6
    max_turns = env.MAX_TURNS / 2
    states_seen = collections.deque(maxlen=max_capacity)
    for i in range(100000):
        if (i % 100) == 0:
            print(i)
        if (i % 100) == 0 and i > 0:
            buffer_stats(states_seen)
        state = env.reset()
        if i > 0 and np.random.uniform(0, 1) > reset_prob:
            flag, sampled_state = sample_from_buffer(states_seen)
            if flag:
                state = sampled_state
        done = False
        while not done:
            state, reward, done, info = env.step(np.random.randint(9))
            states_seen.append(state)

