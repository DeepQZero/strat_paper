from stable_baselines3 import PPO
from exp_2_5_env import FuelPenEnv

def learn_model(model_name):
    env = FuelPenEnv()
    env.log_increment = 100
    model = PPO("MlpPolicy", env, verbose=1, ent_coef=0.01)
    model.learn(total_timesteps=int(2e5), tb_log_name="Passive_Collapse")
    model.save(model_name)

def main():
    model_name = "passive_a2c_model"
    learn_model(model_name)

if __name__ == "__main__":
    main()