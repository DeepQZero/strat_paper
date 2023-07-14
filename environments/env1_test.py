from experiment_1_env import Env
import numpy as np
import matplotlib.pyplot as plt


def main():
    ### Gather States
    env = Env()
    all_states = []
    state = env.reset()
    all_states.append(state)
    done = False
    counter = 0
    while not done:
        counter += 1
        print(counter)
        action = np.array([0, 0])
        state, reward, done, info = env.step(action)
        all_states.append(state)

    ### Plot States
    xs = []
    ys = []
    for state in all_states:
        xs.append(state[0])
        ys.append(state[1])
    plt.scatter(xs, ys)
    plt.show()


if __name__ == "__main__":
    main()
