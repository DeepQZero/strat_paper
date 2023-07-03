from gymnasium import RewardWrapper

class SparseReward(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.reward_threshold = 4.0

    def reward(self, r):
        if r < self.reward_threshold:
            return 0.0
        return r

