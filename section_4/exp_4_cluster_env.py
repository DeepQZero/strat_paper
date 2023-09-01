from multiprocessing import Pool

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import gymnasium as gym

from exp_4_space_env import Env
from lib import dynamics as dyn

class ClusterEnv(gym.Env):
    def __init__(self, cluster_epis=1000, num_clusters=50, capture_radius=1e6, add_fuel_penalty=True, dense_reward=True,
                 drifting=True, capture_reward=True):
        self.env = Env(capture_radius=capture_radius, add_fuel_penalty=add_fuel_penalty, dense_reward=dense_reward,
                       drifting=drifting, capture_reward=capture_reward)
        self.state_buffer = []
        self.CLUSTER_EPIS = cluster_epis
        self.NUM_CLUSTERS = num_clusters
        self.clusters = []
        self.num_resets = 0
        self.fig_counter = 0
        self.state = None
        self.current_trajectory = []
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, seed=None, options=None):
        if (self.num_resets % self.CLUSTER_EPIS) == 0:
            self.cluster()
        if len(self.clusters) == 0:
            state, _ = self.env.reset()
        else:
            if np.random.uniform(0, 1) < 0.5:
                state, _ = self.env.reset()
            else:
                start_state = self.sample_start_state()
                state, _ = self.env.det_reset_helper(start_state)
        state = self.env.det_obs_1()
        self.num_resets += 1
        self.state = state
        self.current_trajectory = []
        return self.det_obs(self.state), None

    def det_obs(self, state) -> np.ndarray:
        """Returns observation per Gym standard."""
        # distance from GEO, velocity vector, angle from enemy base
        # rotate observations so the enemy base is always at zero
        # TODO: rotate if something doesn't work. Maybe rendering
        mobile_pos = (state[0:2] - dyn.GEO) / 5e6
        mobile_vel = (state[2:4]) / dyn.BASE_VEL_Y
        enemy_pos  = (state[8:10] - dyn.GEO) / 5e6
        return np.concatenate((mobile_pos, mobile_vel, enemy_pos, [state[12]], [state[13]]))

    def step(self, action):
        state, reward, done, _, info = self.env.step(action)
        state = self.env.det_obs_1()
        self.filter_state(state)
        self.state = state
        self.current_trajectory.append(state)
        if self.env.is_capture():
            print("CAPTURE")
            #print(self.current_trajectory)
        return self.det_obs(self.state), reward, done, False, info

    def hard_reset(self):
        self.state_buffer = []
        self.clusters = []
        self.current_trajectory = []
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
        if in_zone and good_fuel and good_turn:
            #print("STATE HAS BEEN APPENDED")
            distance = dyn.vec_norm(state[0:2] - state[8:10])
            #print(angle_left_proportion, fuel_left_proportion, turn_left_proportion, distance)
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
        if len(self.state_buffer) < self.NUM_CLUSTERS:
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
                    #print(pos, ang)
                    big_list1.append(pos)
                    big_list2.append(ang)
            plt.scatter(big_list1, big_list2)
            # plt.show()
            plt.savefig('pic' + str(self.fig_counter) + '.jpg')
            plt.close()
            self.fig_counter += 1

    def is_capture(self):
        return self.env.is_capture()

def main():
    env = ClusterEnv()
    env.hard_reset()
    for i in range(int(1e5)):
        if (i % 100) == 0:
            print(i)
        _, _ = env.reset()
        done = False
        while not done:
            if np.random.uniform(0, 1) < 0.0:
                rand_act = np.array([0.0, 0.0])
            else:
                thrust = 10
                rand_thrust = np.random.uniform(0, thrust)
                rand_angle = np.random.uniform(0, 2 * np.pi)
                rand_act = np.array([np.cos(rand_angle), np.sin(rand_angle)]) * rand_thrust
            state, reward, done, _, info = env.step(rand_act)
            if dyn.vec_norm(state[0:2]-state[8:10]) < 5e5:
                print("CAPTURE", state)


if __name__ == "__main__":
    main()

