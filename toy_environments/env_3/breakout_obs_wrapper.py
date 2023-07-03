import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box, Discrete


class CNNObservation(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(low=0, high=255.0, shape=(210, 160, 1), dtype=np.uint8)
        self.action_space = Discrete(4)

    def observation(self, obs):
        return obs.reshape((210, 160, 1))
