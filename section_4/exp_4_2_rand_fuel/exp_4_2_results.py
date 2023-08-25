import pickle
import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np

from exp_4_space_env import Env

def main():
    env = Env(add_fuel_penalty=True, dense_reward=True, capture_radius=5e5)
    NUM_EPISODES = 5000
    THRUST = 10.0
    i = 0
    done = False
    reward_stats = []
    num_captures = 0
    env.reset()
    while i < NUM_EPISODES:
        if (i % 100) == 0:
            print(i)
        while not done:
            action = np.random.uniform(0, THRUST, 2)
            state, reward, done, _, info = env.step(action)
            reward_stats.append(reward)
            if env.is_capture():
                print("CAPTURE")
                num_captures += 1
        state, _ = env.reset()
        done = False
        i += 1
    print("Average reward: ", np.average(reward_stats))
    exp_data = {"reward_stats": reward_stats, "num_captures": num_captures}
    pickle.dump(exp_data, open("exp_4_2_data.pkl", "wb"))
    return

if __name__ == "__main__":
    main()