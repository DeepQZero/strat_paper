import numpy as np
import matplotlib.pyplot as plt
import lib.dynamics as dynamics

import environments.env_1.racetrack


def plot_states(states):
    xs, ys = [], []
    for state in states:
        angle = np.arctan2(state[9], state[8])
        agent_state = state[0:2]
        xs.append(agent_state[0])
        ys.append(agent_state[1])

    circle1 = plt.Circle((0, 0), 6.36e6, color='b')
    circle2 = plt.Circle((-dynamics.GEO, 0), 1e6, color='black', fill=None)
    circle3 = plt.Circle((dynamics.GEO, 0), 1e6, color='black', fill=None)

    fig, ax = plt.subplots()
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.add_patch(circle3)

    plt.xlim([-1e8, 1e8])
    plt.ylim([-1e8, 1e8])
    plt.scatter(xs, ys, c='r', alpha=np.linspace(0.2, 0.8, len(states)))
    plt.colorbar()
    plt.show()


env = environments.env_1.racetrack.Env()
for i in range(10000):
    states = []
    state = env.reset()
    states.append(state)
    done = False
    while not done:
        if np.random.uniform(0, 1) < 1:
            action = np.random.randint(9)
        else:
            action = 4
        state, rew, done, info = env.step(action)
        states.append(state)
    if state[12] == 1:
        plot_states(states)


