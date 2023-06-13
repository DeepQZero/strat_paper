from environments.env_1 import racetrack
from lib import dynamics

import numpy as np

GEO = dynamics.GEO
BASE_VEL_Y = dynamics.BASE_VEL_Y
lower_threshold = GEO - 1e6
upper_threshold = GEO + 1e6

def norm_state(state):
    return GEO * (state[0]**2 + state[1]**2)**0.5

def norm_vel(state):
    return BASE_VEL_Y * (state[2]**2 + state[3]**2)**0.5

def main():
    success_counter = 0
    env = racetrack.Env()
    for i in range(100):
        state = env.reset()
        done = False
        once_below = False
        once_above = False
        j = 0
        while not done:
            j += 1
            action = np.random.randint(9)
            next_state, reward, done, info, = env.step(action)
            print(norm_vel(state))
            if norm_state(next_state) < lower_threshold and not once_below:
                once_below = True
                print("Went below: ", j)
            if norm_state(next_state) > upper_threshold and not once_above:
                once_above = True
                print("Went above: ", j)
            if once_below and norm_state(next_state) >= GEO:
                done = True
                success_counter += 1
                print("Back to GEO: ", j)
            if once_above and norm_state(next_state) <= GEO:
                done = True
                success_counter += 1
                print("Back to GEO: ", j)
            state = next_state
            #print(norm_state(state) / GEO)
        print(i, success_counter, (once_below or once_above), j)
    print(success_counter)
    return success_counter

if __name__ == "__main__":
    num_success = main()
