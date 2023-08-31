import pickle

from stable_baselines3 import PPO
from exp_2_4_env import SparseEnv

def learn_model(model_name, pickle_name):
    env = SparseEnv(log_increment=5e3, log_time=1e5)
    env.PICKLE_NAME = pickle_name
    model = PPO("MlpPolicy", env, verbose=1, ent_coef=0.01)
    model.learn(total_timesteps=int(2e5), tb_log_name="./Passive_Collapse/")
    model.save(model_name)

def main():
    model_name = "passive_a2c_model"
    pickle_files = []
    for i in range(5):
        pickle_files.append("exp_2_4_data_" + str(i) + ".pkl")
    for fname in pickle_files:
        print(fname)
        learn_model(model_name + "_" + str(i), fname)

def process_results():
    pickle_files = []
    for i in range(5):
        pickle_files.append("exp_2_4_data_" + str(i) + ".pkl")
    result_list = []
    for i, fname in enumerate(pickle_files):
        data = pickle.load(open(fname, "rb"))
        if i == 0:
            result_list = data
        else:
            for j, result in enumerate(data):
                result_list[j][1] = result_list[j][1] + result[1]
    pickle.dump(result_list, open("exp_2_4_data_avg.pkl", "wb"))

if __name__ == "__main__":
    process_results()