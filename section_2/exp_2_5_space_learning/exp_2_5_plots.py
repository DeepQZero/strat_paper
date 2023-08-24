# Line graph of the magnitude of the action - continuous actions, no stacked density plots
import pickle

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
palette = sns.cubehelix_palette(reverse=True)

def main():
    result_hist = pickle.load(open("exp_2_5_data.pkl", "rb"))[:25]
    result_hist = np.array(result_hist)
    result_df = pd.DataFrame(result_hist, columns=["x", "y"])
    print(result_df)
    plot = sns.lineplot(x=result_df["x"], y=result_df["y"], palette=palette, hue=1)
    plot.set(xlabel="Timestep", ylabel="Mean Action Magnitude", title="Passive Collapse due to Fuel Penalty")
    plt.grid()
    plt.savefig("2_5_fig.png")
    plt.show()
    return

if __name__ == "__main__":
    main()