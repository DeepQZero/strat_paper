import numpy as np
import matplotlib.pyplot as plt
import lib.dynamics as dynamics

import environments.env_1.racetrack_cont


def plot_states(states):
    xs, ys = [], []
    for state in states:
        # angle = np.arctan2(state[9], state[8])
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


env = environments.env_1.racetrack_cont.Env()
for i in range(1):
    states = []
    state = env.reset()
    states.append(state)
    done = False
    while not done:
        if np.random.uniform(0, 1) < 0:
            action = np.random.uniform(-np.pi, np.pi, 2)
        else:
            action = np.array([0, 0])
        state, rew, done, info = env.step(action)
        print(state)
        states.append(state)
    plot_states(states)


