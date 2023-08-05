import numpy as np
from exp_4_1_env import Env
from lib import dynamics as dyn

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


class ClusterEnv:
    def __init__(self, cluster_epis=100, num_clusters=50):
        self.env = Env()
        self.state_buffer = []
        self.CLUSTER_EPIS = cluster_epis
        self.NUM_CLUSTERS = num_clusters
        self.clusters = []
        self.num_resets = 0
        self.fig_counter = 0

    def reset(self):
        if (self.num_resets % 100) == 0 and self.num_resets > 0:
            self.cluster()
        if len(self.clusters) == 0:
            state = self.env.reset()
        else:
            if np.random.uniform(0, 1) < 0.5:
                state = self.env.reset()
            else:
                start_state = self.sample_start_state()
                state = self.env.det_reset_helper(start_state)
        self.num_resets += 1
        return state

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.filter_state(state)
        return state, reward, done, info

    def hard_reset(self):
        self.state_buffer = []
        self.clusters = []
        self.num_resets = 0

    def filter_state(self, state):
        angle = dyn.abs_angle_diff(state[0:2], state[8:10])
        dist = dyn.vec_norm(state[0:2])
        in_zone = abs(dist - dyn.GEO) < 5e6
        angle_left_proportion = angle/np.pi
        fuel_left_proportion = (self.env.MAX_FUEL - state[13]) / (self.env.MAX_FUEL)
        turn_left_proportion = (self.env.MAX_TURNS - state[12]) / self.env.MAX_TURNS
        good_fuel = angle_left_proportion < fuel_left_proportion
        good_turn = angle_left_proportion < turn_left_proportion
        # print(angle_left_proportion, fuel_left_proportion, turn_left_proportion)
        # if in_zone and good_fuel and good_turn:
        #     self.state_buffer.append(state)
        if in_zone and good_fuel and good_turn:
            self.state_buffer.append(state)

    def sample_start_state(self):
        cluster_choice = np.random.randint(len(self.clusters))
        state_choice = np.random.randint(len(self.clusters[cluster_choice]))
        return self.clusters[cluster_choice][state_choice]

    def cluster(self):
        print("CLUSTERING")
        candidate_states = np.zeros((len(self.state_buffer), 2))
        for i, state in enumerate(self.state_buffer):
            candidate_states[i, 0] = (dyn.vec_norm(state[0:2]) - dyn.GEO) / 10e6
            candidate_states[i, 1] = dyn.abs_angle_diff(state[0:2], state[8:10]) / np.pi
        if candidate_states.shape[0] < self.NUM_CLUSTERS:
            self.clusters = []
            print("FAILED TO FIND ANY STATES")
        else:
            clusterer = KMeans(n_clusters=self.NUM_CLUSTERS)
            indices = clusterer.fit_predict(candidate_states)
            cluster_list = [[] for _ in range(max(indices) + 1)]
            for i, state in enumerate(self.state_buffer):
                cluster_list[indices[i]].append(state)
            self.clusters = cluster_list
            self.state_buffer = []

            big_list1 = []
            big_list2 = []
            for cluster in self.clusters:
                for s in cluster:
                    pos = (dyn.vec_norm(s[0:2]) - dyn.GEO) / 10e6
                    ang = dyn.abs_angle_diff(s[0:2], s[8:10]) / np.pi
                    print(pos, ang)
                    big_list1.append(pos)
                    big_list2.append(ang)
            plt.scatter(big_list1, big_list2)
            # plt.show()
            plt.savefig('pic' + str(self.fig_counter) + '.jpg')
            plt.close()
            self.fig_counter += 1


if __name__ == '__main__':
    env = ClusterEnv()
    env.hard_reset()
    for _ in range(int(1e5)):
        _ = env.reset()
        done = False
        while not done:
            if np.random.uniform(0, 1) < 0.0:
                rand_act = np.array([0.0, 0.0])
            else:
                thrust = 2
                rand_thrust = np.random.uniform(0, thrust)
                rand_angle = np.random.uniform(0, 2 * np.pi)
                rand_act = np.array([np.cos(rand_angle), np.sin(rand_angle)]) * rand_thrust
            state, reward, done, info = env.step(rand_act)
            if dyn.vec_norm(state[0:2]-state[8:10]) < 1e5:
                print("CAPTURE", state)
