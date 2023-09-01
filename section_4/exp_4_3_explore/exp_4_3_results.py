import pickle
import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np

from lib import dynamics as dyn
from exp_4_space_env import Env


def main():
    NUM_STATES = 100
    i = 0
    env = Env(drifting=True)
    state_buffer = []
    while len(state_buffer) < NUM_STATES:
        if (i % 100) == 0:
            print(i)
        i += 1
        done = False
        state, _ = env.reset()
        while not done:
            action = np.random.uniform(0, 10, 2)
            # IF STATEMENT DRIFTING
            #for u in [4, 5, 6, 7]:
            #    if (((env.MAX_FUEL - env.det_obs_1()[13]) / env.MAX_FUEL) < ((8 - u) * 0.125) + 0.02) and (
            #            dyn.abs_angle_diff(env.det_obs_1()[0:2], env.det_obs_1()[8:10]) > (u * np.pi / 8)):
            #        action = np.array([0.0, 0.0])
            state, reward, done, _, _ = env.step(action)
            if env.eval_state(env.det_obs_1()):
                print("STATE HAS BEEN APPENDED")
                state_buffer.append(env.det_obs_1())
    pickle.dump(state_buffer, open("exp_4_3_data.pkl", "wb"))


if __name__ == "__main__":
    main()