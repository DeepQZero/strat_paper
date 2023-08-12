from multiprocessing import Pool
import random

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import gymnasium as gym

from exp_4_2_env import Env
from lib import dynamics as dyn


class DataCollector:
    def __init__(self):
        self.start_buffer = []
        self.capture_buffer = [] # TODO: IMPLEMENT CAPTURE BUFFER
        # Save the intermediate states from the start state to the capture
        # Can we learn from using one winning trajectory as start states? - no clustering, no capture buffer
        # Pickle the start state buffer, unpickle it later
        self.MAX_FUEL = 125.0
        self.MAX_TURNS = 112

    def buffer_sample(self):
        if np.random.uniform(0, 1) < 0.5 and len(self.start_buffer) > 0:
            return self.sample_start_state()
        return None

    def sample_start_state(self):
        return random.choice(self.start_buffer)

    def filter_state(self, state):
        angle = dyn.abs_angle_diff(state[0:2], state[8:10])
        dist = dyn.vec_norm(state[0:2])
        in_zone = abs(dist - dyn.GEO) < 5e6
        angle_left_proportion = angle / np.pi
        fuel_left_proportion = (self.MAX_FUEL - state[13]) / (self.MAX_FUEL)
        turn_left_proportion = (self.MAX_TURNS - state[12]) / self.MAX_TURNS
        good_fuel = angle_left_proportion < fuel_left_proportion
        good_turn = angle_left_proportion < turn_left_proportion
        # print(angle_left_proportion, fuel_left_proportion, turn_left_proportion)
        # if in_zone and good_fuel and good_turn:
        #     self.state_buffer.append(state)
        if in_zone and good_fuel and good_turn:
            print("STATE HAS BEEN APPENDED")
            distance = dyn.vec_norm(state[0:2] - state[8:10])
            print(angle_left_proportion, fuel_left_proportion, turn_left_proportion, distance)
            self.start_buffer.append(state)

    def choose_action(self, state):
        #if np.random.uniform(0, 1) < 0.0 or state[13] > self.MAX_FUEL:
        #    rand_act = np.array([0.0, 0.0])
        for u in [4, 5, 6, 7]:
            if (((self.MAX_FUEL - state[13]) / self.MAX_FUEL) < ((8-u)*0.125)+0.02) and (dyn.abs_angle_diff(state[0:2], state[8:10]) > (u*np.pi/8)):
                return np.array([0.0, 0.0])
        thrust = 10
        rand_thrust = np.random.uniform(0, thrust)
        rand_angle = np.random.uniform(0, 2 * np.pi)
        rand_act = np.array([np.cos(rand_angle), np.sin(rand_angle)]) * rand_thrust
        return rand_act

def main():
    data_collector = DataCollector()
    env = Env()
    for i in range(int(1e5)):
        if (i % 100) == 0:
            print(i)
        if np.random.uniform(0, 1) < 0.5:
            state = data_collector.buffer_sample()
            if state is None:
                state, _ = env.reset()
            else:
                env.det_reset_helper(state)
        else:
            state, _ = env.reset()
        done = False
        while not done:
            rand_act = data_collector.choose_action(state)
            state, reward, done, _, info = env.step(rand_act)
            data_collector.filter_state(state)
            # if dyn.vec_norm(state[0:2]-state[8:10]) < 1e5:
            # print("CAPTURE", state)


if __name__ == "__main__":
    main() # TODO: Parallelize the main function

