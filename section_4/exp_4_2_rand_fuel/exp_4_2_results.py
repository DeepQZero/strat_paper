import pickle
import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np

from exp_4_space_env import Env

def main():
    env = Env(add_fuel_penalty=True, dense_reward=True, capture_radius=1e6, drifting=False)
    NUM_EPISODES = 5000
    THRUST = 10.0
    i = 0
    done = False
    reward_stats = []
    turn_stats = []
    num_captures = 0
    num_turns = 0
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
            num_turns += 1
        state, _ = env.reset()
        done = False
        i += 1
        turn_stats.append(num_turns)
        num_turns = 0
    print("Average reward: ", np.average(reward_stats))
    print("Average number of turns: ", np.average(turn_stats))
    exp_data = {"reward_stats": reward_stats, "num_captures": num_captures, "num_turns": turn_stats}
    pickle.dump(exp_data, open("exp_4_2_data.pkl", "wb"))
    return

def testing():
    exp_data = pickle.load(open("exp_4_2_data.pkl", "rb"))
    print("Average reward: ", np.average(exp_data["reward_stats"]))
    print("Average number of turns: ", np.average(exp_data["num_turns"]))
    print("Num captures: ", exp_data["num_captures"])

if __name__ == "__main__":
    testing()