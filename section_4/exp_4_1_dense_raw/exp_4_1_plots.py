import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
palette = sns.cubehelix_palette(reverse=True)

from tensorboard.backend.event_processing import event_accumulator

def parse_tensorboard(path, scalars):
    """returns a dictionary of pandas dataframes for each requested scalar"""
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _absorb_print = ea.Reload()
    # make sure the scalars are in the event accumulator tags
    assert all(
        s in ea.Tags()["scalars"] for s in scalars
    ), "some scalars were not found in the event accumulator"
    return {k: pd.DataFrame(ea.Scalars(k)) for k in scalars}

def load_experiment():
    path = r"C:\Users\Kyler\Desktop\GitHub\strat_paper\section_4\exp_4_1_dense_raw\tb_logs\exp_4_1\PPO_1\events.out.tfevents.1692949591.LAPTOP-B318PKJC.19004.0"
    scalars = ["eval/mean_reward", "rollout/ep_rew_mean"]
    exp_dict = parse_tensorboard(path, scalars)
    return exp_dict

def plot_data(data, save_name, ylabel, title, hue=1):
    ax = sns.lineplot(x=data["step"], y=data["value"], palette=palette, hue=hue)
    ax.set(xlabel="Timestep", ylabel=ylabel, title=title)
    plt.grid()
    plt.legend([], [], frameon=False)
    plt.savefig(save_name)
    plt.show()

def main():
    exp_dict = load_experiment()
    eval_data = exp_dict["eval/mean_reward"]
    train_data = exp_dict["rollout/ep_rew_mean"]
    plot_data(train_data, "4_1_fig_1.png", "Mean Training Reward", "PPO Training Data with Dense Rewards")
    plot_data(eval_data, "4_1_fig_2.png", "Mean Eval Reward", "PPO Evaluation Data With Dense Rewards")
    return

if __name__ == "__main__":
    main()