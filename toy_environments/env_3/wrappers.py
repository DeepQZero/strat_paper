import cv2
import gymnasium as gym
import gymnasium.spaces
import numpy as np
import collections
import matplotlib.pyplot as plt
import sys

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
        # img = img.astype(np.uint8)  # TODO Figure out if this is necessary
        return img


# class ScaledFloatFrameWrapper(gym.ObservationWrapper):
#     def __init__(self, env=None):
#         super(ScaledFloatFrameWrapper, self).__init__(env)
#
#     def observation(self, obs):
#         return np.array(obs).astype(np.float32) / 255.0


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
        #return LazyFrames(list(self.frames))


class LazyFrames(object):
    def __init__(self, frames):
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1).reshape((4, 84, 84))
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]


def make_original_dqn(env):
    env = EpisodicLifeWrapper(env)
    env = FireResetWrapper(env)
    env = WarpFrameWrapper(env)
    env = FrameStackWrapper(env, k=4)
    env = ManyActionWrapper(env, k=4)
    env = ClipRewardWrapper(env)
    env = MinimalActionWrapper(env)
    return env


if __name__ == "__main__":
    env = gym.make("ALE/Breakout-v5", render_mode='human', repeat_action_probability=0.00, full_action_space=False, frameskip=1)
    env = EpisodicLifeWrapper(env)
    env = FireResetWrapper(env)
    env = WarpFrameWrapper(env)
    env = FrameStackWrapper(env, k=4)
    # env = ManyActionWrapper(env, k=1)
    # env = ClipRewardWrapper(env)
    # env = MinimalActionWrapper(env)
    obs, info = env.reset()
    # print('size: ', sys.getsizeof(obs))
    done = False
    while not done:
        obs, rew, done1, done2, info = env.step(np.random.randint(4))
        print(obs)
        done = done1 or done2
        if done:
            env.reset()
            done = False
