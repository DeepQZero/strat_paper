import numpy as np

from lib import dynamics as dyn


class Env:  # TODO RENAME SpaceEnv
    def __init__(self, step_len: int = 10800, dis: int = 180,
                 max_turns: int = 8*14, max_fuel=100) -> None:
        self.DIS = dis
        self.UP_LEN = step_len / dis
        self.MAX_TURNS = max_turns
        self.MAX_FUEL = max_fuel
        self.mobile = None
        self.cap_base = None
        self.ret_base = None
        self.time_step = None
        self.total_fuel = None

    def reset(self) -> np.ndarray:
        """Resets environment and returns observation per Gym Standard."""
        self.mobile = np.array([-dyn.GEO, 0.0, 0.0, -dyn.BASE_VEL_Y])
        self.ret_base = np.array([-dyn.GEO, 0.0, 0.0, -dyn.BASE_VEL_Y])
        self.cap_base = np.array([dyn.GEO, 0.0, 0.0, dyn.BASE_VEL_Y])
        self.time_step = 0
        self.total_fuel = 0
        return self.det_obs()

    def det_reset(self, mobile, ret_base, cap_base, time_step, total_fuel) -> np.ndarray:
        """Resets environment deterministically."""
        self.mobile = mobile
        self.ret_base = ret_base
        self.cap_base = cap_base
        self.time_step = time_step
        self.total_fuel = total_fuel
        return self.det_obs()

    def det_obs(self) -> np.ndarray:
        """Returns observation per Gym standard."""
        return np.concatenate((self.mobile, self.ret_base,
                               self.cap_base, [self.time_step]))

    def step(self, action: np.ndarray) -> tuple:
        """Advances environment forward one time step, returns Gym signals."""
        self.mobile[2:4] += action if self.total_fuel < self.MAX_FUEL else np.array([0.0, 0.0])
        self.total_fuel += dyn.vec_norm(action)
        self.mobile = self.prop_unit(self.mobile)
        self.ret_base = self.prop_unit(self.ret_base)
        self.cap_base = self.prop_unit(self.cap_base)
        self.time_step += 1
        return self.det_obs(), self.det_reward(), self.is_done(), {}

    def is_done(self) -> bool:
        """Determines if episode has reached termination."""
        return self.time_step == self.MAX_TURNS

    def det_reward(self) -> float:
        """Returns reward at current time step."""
        if self.is_done():
            return 0.0
        else:
            return 0.0

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
            state, reward, done, info = env.step(np.random.randint(9))


# if __name__ == "__main__":
#     env_test()
