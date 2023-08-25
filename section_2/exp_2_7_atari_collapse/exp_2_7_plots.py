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
        adj_result = ((np.array(result[1:]) - 0.00) / 1.00)[0]
        passive_prob = adj_result[0] + adj_result[1]
        right_prob = adj_result[2]
        left_prob = adj_result[3]
        #adj_result = np.array([passive_prob, right_prob, left_prob])
        adj_result = {"NO OP": passive_prob, "RIGHT": right_prob, "LEFT":left_prob}
        # Combine passive action and fire
        #new_result_hist.append(adj_result)
        new_result_hist[str(timestep) + " steps"] = adj_result
    return new_result_hist


def main():
    result_hist = pickle.load(open("exp_2_7_data.pkl", "rb"))
    new_result_hist = proc_results(result_hist)
    steps_to_plot = [5000, 10000, 20000, 50000]
    print(new_result_hist)
    result_df = dict()
    for s in steps_to_plot:
        key = str(s)+" steps"
        action_dist = new_result_hist[key]
        result_df[key] = action_dist
    result_df = pd.DataFrame(result_df)
    ax = sns.lineplot(result_df, palette=palette)
    line = ax.get_lines()
    #plt.grid(alpha=0.25)
    ax.fill_between(line[0].get_xdata(), line[1].get_ydata(), color=palette[0], alpha=0.6)
    ax.fill_between(line[1].get_xdata(), line[2].get_ydata(), color=palette[1], alpha=0.6)
    ax.fill_between(line[2].get_xdata(), line[3].get_ydata(), color=palette[2], alpha=0.6)
    ax.set(xlabel="Action", ylabel="Probability", title="Passive Collapse with Action Penalties in ATARI Breakout")
    plt.savefig("2_7_fig.png")
    plt.show()

    return

if __name__ == "__main__":
    main()