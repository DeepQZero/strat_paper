import numpy as np

from lib import dynamics as dyn


class Env:
    def __init__(self, step_length=3600, discretization=60, max_turns=24*28):
        self.DISCRETIZATION = discretization
        self.UPDATE_LENGTH = step_length / discretization
        self.MAX_TURNS = max_turns
        self.current_turn = None
        self.unit = None
        self.base = None
        self.return_base = None

    def reset(self) -> np.ndarray:
        """Resets environment. Returns first observation per Gym Standard."""
        self.current_turn = 0
        self.unit = np.array([-dyn.GEO, 0.0, 0.0, -dyn.BASE_VEL_Y])
        self.base = np.array([dyn.GEO, 0.0, 0.0, dyn.BASE_VEL_Y])
        self.return_base = np.array([-dyn.GEO, 0.0, 0.0, -dyn.BASE_VEL_Y])
        return self.det_obs()

    def det_reset(self, turn, unit, base, return_base):
        self.current_turn = turn
        self.unit = unit
        self.base = base
        self.return_base = return_base
        return self.det_obs()

    def det_obs(self) -> np.ndarray:
        """Returns observation by Gym standard."""
        return np.concatenate((self.unit, self.base, self.return_base, [self.current_turn]))

    def step(self, action):
        self.unit[2:4] += action
        self.unit = self.prop_unit(self.unit)
        self.base = self.prop_unit(self.base)
        self.current_turn += 1
        return self.det_obs(), self.det_reward(), self.is_done(), {}

    def is_done(self):
        return self.current_turn == self.MAX_TURNS # or abs(dyn.norm(self.unit[0], self.unit[1]) - dyn.GEO) > dyn.GEO_BOUND

    def det_reward(self):
        if self.is_done():
            return 0
        else:
            return 0

    def prop_unit(self, unit):
        return dyn.propagate(unit[0:4], self.DISCRETIZATION, self.UPDATE_LENGTH)


if __name__ == "__main__":
    env = Env()
    for i in range(100000):
        print(i)
        state = env.reset()
        done = False
        while not done:
            state, reward, done, info = env.step(np.random.randint(9))