import numpy as np
import gymnasium as gym
from lib import dynamics as dyn
# TODO: Try to render the environment - plot positions with a colorbar
# condition the color on time

class Env(gym.Env):
    def __init__(self, step_len: int = 10800, dis: int = 180,
                 max_turns: int = 8*14, max_fuel=125, add_fuel_penalty=True, capture_radius=1e6, noise=True) -> None:
        self.DIS = dis
        self.UP_LEN = step_len / dis
        self.MAX_TURNS = max_turns
        self.MAX_FUEL = max_fuel
        self.FUEL_MULTIPLIER = 0.1
        self.SIGMA = 0.01 if noise else 0.00 # Gaussian noise in actions
        self.CAPTURE_RADIUS = capture_radius
        self.angle_diff = 1.0
        self.mobile = None
        self.cap_base = None
        self.ret_base = None
        self.time_step = None
        self.total_fuel = None
        self.add_fuel_penalty = add_fuel_penalty
        self.observation_space = gym.spaces.Box(-1000, 1000, shape=(8,))
        self.action_space = gym.spaces.Box(0, 10.0, shape=(2,))

    def reset(self, seed=None, options=None) -> tuple:
        """Resets environment and returns observation per Gym Standard."""
        self.mobile = np.array([-dyn.GEO, 0.0, 0.0, -dyn.BASE_VEL_Y])
        self.ret_base = np.array([-dyn.GEO, 0.0, 0.0, -dyn.BASE_VEL_Y])
        self.cap_base = np.array([dyn.GEO, 0.0, 0.0, dyn.BASE_VEL_Y])
        self.time_step = 0
        self.total_fuel = 0
        self.angle_diff = 1.0
        return self.det_obs(), None

    def det_reset(self, mobile, ret_base, cap_base, time_step, total_fuel) -> np.ndarray:
        """Resets environment deterministically."""
        self.mobile = mobile
        self.ret_base = ret_base
        self.cap_base = cap_base
        self.time_step = time_step
        self.total_fuel = total_fuel
        return self.det_obs()

    def det_reset_helper(self, state) -> tuple:
        """Resets environment deterministically."""
        self.mobile = state[0:4]
        self.ret_base = state[4:8]
        self.cap_base = state[8:12]
        self.time_step = state[12]
        self.total_fuel = state[13]
        return self.det_obs(), None

    def det_obs(self) -> np.ndarray:
        """Returns observation per Gym standard."""
        return np.concatenate((self.mobile, self.ret_base,
                               self.cap_base, [self.time_step], [self.total_fuel]))

    def process_action(self, action):
        angle = (action[1] / 5 * np.pi) + np.random.normal(0, self.SIGMA)
        return np.array([action[0]*np.cos(angle), action[0]*np.sin(angle)])

    def step(self, action: np.ndarray) -> tuple:
        """Advances environment forward one time step, returns Gym signals."""
        action = self.process_action(action)
        # DRIFTING
        action = np.array([0.0, 0.0]) if (self.total_fuel + dyn.vec_norm(action) > self.MAX_FUEL) else action
        self.mobile[2:4] += action
        self.total_fuel += dyn.vec_norm(action)
        self.mobile = self.prop_unit(self.mobile)
        self.ret_base = self.prop_unit(self.ret_base)
        self.cap_base = self.prop_unit(self.cap_base)
        self.time_step += 1
        return self.det_obs(), self.det_reward(action), self.is_done(), False, {}

    def is_done(self) -> bool:
        """Determines if episode has reached termination."""
        return self.is_capture() or self.is_timeout() #or self.is_out_of_fuel()

    def is_capture(self) -> bool:
        return dyn.vec_norm(self.mobile[0:2] - self.cap_base[0:2]) < self.CAPTURE_RADIUS

    def is_close_capture(self) -> bool:
        return dyn.vec_norm(self.mobile[0:2] - self.cap_base[0:2]) < 5e6

    def is_timeout(self) -> bool:
        return self.time_step == self.MAX_TURNS

    def is_out_of_fuel(self) -> bool:
        return (self.MAX_FUEL - self.total_fuel) < (1 * dyn.vec_norm(np.array([1.0, 1.0])))

    def det_fuel_rew(self, action) -> float:
        return -1 * dyn.vec_norm(action)/self.MAX_FUEL * self.FUEL_MULTIPLIER

    def det_term_rew(self) -> float:
        if self.is_capture():
            print("CAPTURE")
            return 10.0
        elif self.is_timeout():
            return 0.0
        else:
            return 0.0

    def det_reward(self, action) -> float:
        """Returns reward at current time step."""
        #return self.det_term_rew() + self.det_fuel_rew(action) + self.det_angle_reward()
        angle_reward = self.det_angle_reward()
        if self.add_fuel_penalty:
            return self.det_term_rew() + self.det_fuel_rew(action) + angle_reward
        return self.det_term_rew() #+ angle_reward

    def det_angle_reward(self) -> float:
        angle = dyn.abs_angle_diff(self.mobile[0:2], self.cap_base[0:2])
        new_angle_prop = angle / np.pi
        reward = self.angle_diff - new_angle_prop
        self.angle_diff = new_angle_prop
        return reward

    def prop_unit(self, unit: np.ndarray) -> np.ndarray:
        """Propagates a given unit state forward one time step."""
        return dyn.propagate(unit[0:4], self.DIS, self.UP_LEN)


def env_test():
    env = Env()
    for i in range(100):
        print(i)
        _ = env.reset()
        done = False
        while not done:
            state, reward, done, info = env.step(np.random.uniform(0, 10, 2))


if __name__ == "__main__":
     env_test()