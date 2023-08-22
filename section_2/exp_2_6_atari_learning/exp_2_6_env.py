import cv2
import gymnasium as gym
import numpy as np
import collections

from exp_2_6_plots import ActionDist


class ClipRewardWrapper(gym.RewardWrapper):
    def __init__(self, env=None):
        super(ClipRewardWrapper, self).__init__(env)

    def reward(self, reward):
        return np.sign(reward)


class MinimalActionWrapper(gym.ActionWrapper):
    def __init__(self, env=None):
        super(MinimalActionWrapper, self).__init__(env)

    def action(self, act):
        if act == 0:
            return 0
        elif act == 1:
            return 2
        elif act == 2:
            return 3
        elif act == 3:
            return 1
        else:
            return 0


class WarpFrameWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(WarpFrameWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8)

    def observation(self, obs):
        obs = obs[:, :, 0] * 0.299 + \
              obs[:, :, 1] * 0.587 + \
              obs[:, :, 2] * 0.114
        resized_screen = cv2.resize(obs, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        img = np.moveaxis(x_t, 2, 0)
        return img


class FireResetWrapper(gym.Wrapper):
    def __init__(self, env):
        super(FireResetWrapper, self).__init__(env)

    def reset(self, seed=None, options=None):
        self.env.reset()
        obs, _, _, _, _ = self.env.step(1)
        return obs, None

    def step(self, action):
        return self.env.step(action)


class EpisodicLifeWrapper(gym.Wrapper):
    def __init__(self, env):
        super(EpisodicLifeWrapper, self).__init__(env)
        self.is_game_over = True
        self.current_lives = 0

    def step(self, action):
        obs, rew, done1, done2, info = self.env.step(action)
        done = done1 or done2
        self.is_game_over = done
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.current_lives:
            done = True
        self.current_lives = lives
        return obs, rew, done1, done2, info

    def reset(self, seed=None, options=None):
        if self.is_game_over:
            obs = self.env.reset()
        else:
            obs, _, _, _, _ = self.env.step(0)
        self.current_lives = self.env.unwrapped.ale.lives()
        return obs, None


class ManyActionWrapper(gym.Wrapper):
    def __init__(self, env=None, k=4):
        super(ManyActionWrapper, self).__init__(env)
        self.skip_num = k

    def step(self, action):
        tot_rew = 0.0
        for _ in range(self.skip_num):
            obs, rew, done1, done2, info = self.env.step(action)
            tot_rew += rew
            done = done1 or done2
            if done:
                break
        return obs, tot_rew, done1, done2, info

    def reset(self, seed=None, options=None):
        return self.env.reset()


class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env=None, k=4):
        super(FrameStackWrapper, self).__init__(env)
        self.skip = k
        self.frames = collections.deque([], maxlen=k)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset()
        for _ in range(self.skip):
            self.frames.append(obs)
        return self.concat_obs(), None

    def step(self, action):
        obs, rew, done1, done2, info = self.env.step(action)
        obs = obs.astype(np.uint8)
        self.frames.append(obs)
        return self.concat_obs(), rew, done1, done2, info

    def concat_obs(self):
        frame_list = list(self.frames)
        frame_array = np.array(frame_list)
        frame_array = np.squeeze(frame_array, axis=1)
        return frame_array

class SparseReward(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.reward_threshold = 4.0

    def reward(self, r):
        if r < self.reward_threshold:
            return 0.0
        return r


def make_env(env_name="ALE/Breakout-v5", render=False, reward_wrapped=False):
    if render:
        env = gym.make(env_name, render_mode="human", repeat_action_probability=0.00,
                       full_action_space=False, frameskip=1) # obs_type="grayscale"
    else:
        env = gym.make(env_name, repeat_action_probability=0.00, full_action_space=False,
                       frameskip=1) # obs_type="grayscale"
    env = EpisodicLifeWrapper(env)
    env = FireResetWrapper(env)
    env = WarpFrameWrapper(env)
    env = FrameStackWrapper(env, k=4) # k = 4
    env = ManyActionWrapper(env, k=4) # k = 4
    env = ClipRewardWrapper(env)
    env = MinimalActionWrapper(env)
    if reward_wrapped:
        env = SparseReward(env)
    env = ActionDist(env)
    return env