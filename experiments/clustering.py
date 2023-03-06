import numpy as np
import matplotlib.pyplot as plt
import os
from cluster_wrapper import ClusterEnvironment


def action_decoder(action):
    if action == 0:
        return 1, 1
    elif action == 1:
        return 1, 0
    elif action == 2:
        return 1, -1
    elif action == 3:
        return 0, 1
    elif action == 4:
        return 0, -1
    elif action == 5:
        return -1, 1
    elif action == 6:
        return -1, 0
    elif action == 7:
        return -1, -1
    elif action == 8:
        return 0, 0


def plot_centers(centers):
    xs, ys, cs = [], [], []
    for i, center in enumerate(clusters):
        r, theta = center
        x, y = r * np.cos(theta), r * np.sin(theta)
        xs.append(x)
        ys.append(y)
        cs.append(i)

    figure, axes = plt.subplots()
    Drawing_uncolored_circle1 = plt.Circle((0.0, 0.0), 50, fill=False)
    Drawing_uncolored_circle2 = plt.Circle((0.0, 0.0), 60, fill=False)
    axes.set_aspect(1)
    axes.add_artist(Drawing_uncolored_circle1)
    axes.add_artist(Drawing_uncolored_circle2)
    plt.xlim([-65, 65])
    plt.ylim([-65, 65])
    plt.scatter(xs, ys, c='r')
    plt.colorbar()
    plt.show()


env = ClusterEnvironment()
os.mkdir("cluster_pictures")
for i in range(1000):
    states = []
    for ii in range(1000):
        state = env.reset()
        states.append(state)
        done = False
        while not done:
            if np.random.uniform(0, 1) < 0.1:
                action = np.random.randint(9)
            else:
                action = 8
            new_action = action_decoder(action)
            state, rew, done, info = env.step(new_action)
            states.append(state)
            print(state)
    env.cluster()
    clusters = env.clusters
    if (i % 10) == 0:
        plot_centers(clusters)


