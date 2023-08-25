import gc

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback

from exp_2_6_env import make_env


def test_model(model_name, reward_wrapped, num_episodes=int(1e1)):
    env = make_env(render=True, reward_wrapped=reward_wrapped)
    model = DQN.load(model_name, env=env)
    model.device = "cpu"
    state = env.reset()[0]
    episodes_completed = 0
    while episodes_completed < num_episodes:
        action, _states = model.predict(state)
        state, reward, done1, done2, info = env.step(action)
        if done1 or done2:
            episodes_completed += 1
            state = env.reset()[0]


def learn_model(model_name, num_timesteps=int(1e6)):
    env = make_env(render=False, reward_wrapped=True)
    tb_log_path = r"./" + model_name + r"_tensorboard/"
    # Normalizes images by default - see docs
    model = DQN("CnnPolicy", env, verbose=1, device="cuda", tensorboard_log=tb_log_path, learning_starts=0,
                exploration_fraction=0.1, exploration_initial_eps=0.01, exploration_final_eps=0.01, batch_size=32)
    eval_env = make_env(render=False, reward_wrapped=True)
    eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/",
                             log_path="./logs/", eval_freq=int(1e4), n_eval_episodes=100)
    model.learn(total_timesteps=num_timesteps)#, callback=[eval_callback])
    model.save(model_name)
    del model # delete the model object/replay buffer from RAM. Model file remains stored.
    gc.collect() # Clear out the RAM after training is done so models can be trained in succession.


def main():
    learn_model("exp_2_6", num_timesteps=int(2e6))
    return

if __name__ == "__main__":
    main()
