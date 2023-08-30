# Stacked density plots - discrete actions
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

palette = sns.cubehelix_palette(reverse=True)

# Subtract out the random exploration in the results.
def proc_results(result_hist):
    new_result_hist = dict()
    for result in result_hist:
        timestep = result[0]
        adj_result = ((np.array(result[1:]) - 0.01) / 0.96)[0]
        passive_prob = adj_result[0]# + adj_result[1]
        no_op_prob = adj_result[1]
        right_prob = adj_result[2]
        left_prob = adj_result[3]
        #adj_result = np.array([passive_prob, right_prob, left_prob])
        adj_result = {"PASSIVE": passive_prob, "FIRE": no_op_prob, "RIGHT": right_prob, "LEFT":left_prob}
        # Combine passive action and fire
        #new_result_hist.append(adj_result)
        new_result_hist[timestep] = adj_result
    return new_result_hist

def stacked_density_plot():
    result_hist = pickle.load(open("exp_2_7_data.pkl", "rb"))
    new_result_hist = proc_results(result_hist)
    passive = dict()
    fire = dict()
    right = dict()
    left = dict()
    for key in new_result_hist.keys():
        if key > 50000:
            break
        passive[key] = new_result_hist[key]["PASSIVE"] + new_result_hist[key]["FIRE"] + new_result_hist[key]["RIGHT"] + new_result_hist[key]["LEFT"]
        fire[key] = new_result_hist[key]["FIRE"] + new_result_hist[key]["RIGHT"] + new_result_hist[key]["LEFT"]
        right[key] = new_result_hist[key]["RIGHT"] + new_result_hist[key]["LEFT"]
        left[key] = new_result_hist[key]["LEFT"]
    result_df = pd.DataFrame({"PASSIVE":passive, "FIRE":fire, "RIGHT": right, "LEFT": left})
    ax = sns.lineplot(result_df, palette=palette, linewidth=2, dashes=False)
    line = ax.get_lines()
    ax.fill_between(line[0].get_xdata(), line[0].get_ydata(), color=palette[0], alpha=1.0)
    ax.fill_between(line[0].get_xdata(), line[1].get_ydata(), color=palette[1], alpha=1.0)
    ax.fill_between(line[0].get_xdata(), line[2].get_ydata(), color=palette[2], alpha=1.0)
    ax.fill_between(line[0].get_xdata(), line[3].get_ydata(), color=palette[3], alpha=1.0)
    ax.set(xlabel="Timestep", ylabel="Action Density", title="Passive Action Collapse in ATARI Breakout")
    plt.savefig("2_7_fig.png")
    plt.show()
    return

if __name__ == "__main__":
    stacked_density_plot()

