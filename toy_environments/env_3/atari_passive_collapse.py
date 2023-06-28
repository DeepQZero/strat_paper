import numpy as np
import gym
import torch

from stable_baselines3 import A2C

from breakout_reward_wrapper import SparseReward

def test_model(model_name):
    env = make_env(model_name)
    model = A2C.load(model_name)
    state = env.reset()

    while True:
        dist = model.policy.get_distribution(torch.unsqueeze(torch.tensor(state), 0))
        log_prob = dist.log_prob(torch.tensor(list(range(9))))
        prob = torch.exp(log_prob)
        print(prob)
        action, _states = model.predict([state])
        state, reward, done, info = env.step(action)
        # print(state, action, done)

def learn_model(model_name):
    env = make_env()
    env.reset()
    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=int(1e6))
    model.save(model_name)

def make_env():
    env_name = "ALE/Breakout-v5"
    env = gym.make(env_name, render_mode="human")
    env = SparseReward(env)
    return env


def main():
    learn_model("breakout_A2C")
    #test_model("breakout_A2C")
    return


if __name__ == "__main__":
    main()
