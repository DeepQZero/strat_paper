import random

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import gym
from gym import spaces
from gym.spaces import Box, Tuple, Discrete

from fuel_env import Environment


class ClusterEnvironment(gym.Env):
    def __init__(self):
        self.clusters = [(55, 5/360*2*np.pi)]*10
        self.points = []
        self.num_episodes = 0
        self.env = Environment()
        self.EPOCH = 100
        self.RECORD = True  # TODO Reward penalty or terminal timestep
        self.action_space = Discrete(9)
        self.observation_space = Box(0, 62, shape=(2,))

    def hard_reset(self):
        self.RECORD = True
        self.num_episodes = 0
        self.clusters = [(55, 5/360*2*np.pi)]*10

    def exploit_reset(self):
        self.RECORD = False
        return self.env.reset((55, 5/360*2*np.pi))

    def reset(self):
        self.RECORD = True
        self.num_episodes += 1
        if (self.num_episodes % self.EPOCH) == 0:
            self.cluster()
            self.plot_centers(self.clusters)
        start = random.choice(self.clusters)
        return self.env.reset(start)


    def step(self, action):
        #new_action = self.action_decoder(action)
        state, rew, done, info = self.env.step(action)
        r, theta = state
        x, y = r*np.cos(theta), r*np.sin(theta)
        if not done and self.RECORD:
            self.points.append((x, y))
        # return np.array([(x-55)/10, (y-55)/10]), rew, done, info
        return np.array([(r-55)/10, theta]), rew, done, info

    def cluster(self, num_clusters=50):
        clusterer = KMeans(n_clusters=num_clusters, n_init=10)
        clusterer.fit(self.points)
        temp = [(55, 5/360*2*np.pi)]
        for x, y in clusterer.cluster_centers_:
            temp.append((np.sqrt(x**2 + y**2), np.arctan2(y, x) % (2 * np.pi)))
        self.clusters = temp
        self.points = []

    def action_decoder(self, action):
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

    def plot_centers(self, clusters):
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
        plt.savefig('cluster_pictures/pic_' + str(int(self.num_episodes/self.EPOCH)))
        plt.close()
