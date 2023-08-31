import pickle

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
palette = sns.cubehelix_palette(reverse=True)


def ridgeplot():
    result_hist = pickle.load(open("exp_2_4_data_avg.pkl", "rb"))[:25]
    result_dict = dict(timestep=[], data=[])
    for result in result_hist:
        if result[0] in [10000, 20000, 30000, 40000, 50000]:
            for action in result[1]:
                result_dict["timestep"].append(result[0])
                result_dict["data"].append(action)
    result_df = pd.DataFrame(result_dict)
    print(result_df)
    g = sns.FacetGrid(result_df, row="timestep", palette=palette, aspect=5, height=1, hue="timestep")
    g.map_dataframe(sns.kdeplot, x="data", fill=True, alpha=1.0) # bw_adjust=1
    g.map_dataframe(sns.kdeplot, x="data", color='black')
    g.map(plt.axhline, y=0, lw=2, clip_on=False)
    g.fig.subplots_adjust(hspace=-0.5)
    g.set(xticks=np.arange(0.0, 16, 2.5), yticks=[])
    g.despine(bottom=True, left=True)
    g.set(xlabel="Action Magnitude Distribution", ylabel="")
    g.fig.tight_layout()
    plt.savefig("2_4_fig.png")
    plt.show()
    return

if __name__ == "__main__":
    ridgeplot()