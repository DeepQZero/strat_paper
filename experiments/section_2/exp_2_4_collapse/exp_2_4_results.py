from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import EvalCallback

from exp_2_4_env import Env

def learn_model(model_name):
    env = Env()
    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=50000)
    model.save(model_name)

def main():
    model_name = "passive_a2c_model"
    learn_model(model_name)

if __name__ == "__main__":
    main()