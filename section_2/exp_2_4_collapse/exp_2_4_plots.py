# Line graph of the magnitude of the action - continuous actions, no stacked density plots
import pickle

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
palette = sns.cubehelix_palette(reverse=True)

def lineplot():
    result_hist = pickle.load(open("exp_2_4_data.pkl", "rb"))[:25]
    result_hist = np.array(result_hist)
    result_df = pd.DataFrame(result_hist, columns=["x", "y"])
    plot = sns.lineplot(x=result_df["x"], y=result_df["y"], palette=palette, hue=1)
    plot.set(xlabel="Timestep", ylabel="Mean Action Magnitude", title="No Policy Improvement With Sparse Rewards")
    plt.grid()
    plt.savefig("2_4_fig.png")
    plt.show()
    return

def ridgeplot():
    result_hist = pickle.load(open("exp_2_4_data.pkl", "rb"))[:25]
    result_dict = dict(tstep=[], data=[])
    for result in result_hist:
        if result[0] in [10000, 20000, 30000, 40000, 50000]:
            for action in result[1]:
                result_dict["tstep"].append(result[0])
                result_dict["data"].append(action)
    result_df = pd.DataFrame(result_dict)
    print(result_df)
    g = sns.FacetGrid(result_df, row="tstep", palette=palette, aspect=10, height=0.5, hue="tstep")
    g.map_dataframe(sns.kdeplot, x="data", fill=True, alpha=1.0, bw_adjust=1)
    g.map_dataframe(sns.kdeplot, x="data", color='black', bw_adjust=1)
    g.map(plt.axhline, y=0, lw=2, clip_on=False)
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, "", fontweight="bold", color=color, ha="left", va="center", transform=ax.transAxes)

    g.map(label, "tstep")
    g.fig.subplots_adjust(hspace=-0.25)
    g.set_titles("")
    g.set(yticks=[])
    g.despine(left=True)
    g.set(ylabel="")
    g.tight_layout()
    plt.savefig("2_4_fig.png")
    plt.show()
    return

if __name__ == "__main__":
    ridgeplot()