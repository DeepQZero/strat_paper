import pickle
import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
palette = sns.cubehelix_palette(reverse=True, as_cmap=True)

from lib import dynamics as dyn
from exp_4_cluster_env import ClusterEnv
from exp_4_4_results import DataCollector

def proc_results(cluster_list):
    result_df = pd.DataFrame(columns=["mobile_pos_x", "mobile_pos_y", "mobile_vel", "capture_pos", "capture_vel",
                                      "return_pos", "return_vel", "turn_num", "total_fuel", "cluster_id"])
    NUM_STATES = 250
    for i, cluster in enumerate(cluster_list):
        for k, state in enumerate(cluster):
            if k >= NUM_STATES:
                break
            mobile_pos = state[0:2]
            mobile_vel = state[2:4]
            capture_pos = state[4:6]
            capture_vel = state[6:8]
            return_pos = state[8:10]
            return_vel = state[10:12]
            turn_num = (112 - state[12]) * 100/112
            total_fuel = (125 - state[13])*100/125  # Remaining fuel, not fuel used. For color grading

            # Rotate the frame so the capture position is always at zero
            angle = float(dyn.abs_angle_diff(mobile_pos, capture_pos))
            mobile_pos = dyn.vec_rotate(mobile_pos, angle)

            result_dict = {"mobile_pos_x": mobile_pos[0], "mobile_pos_y": mobile_pos[1], "mobile_vel": mobile_vel,
                           "capture_pos": capture_pos, "capture_vel": capture_vel,
                           "return_pos": return_pos, "return_vel": return_vel,
                           "turn_num": turn_num, "total_fuel": total_fuel, "cluster_id": i}
            result_df.loc[len(result_df.index)] = result_dict
    return result_df

def main():
    data_collector = pickle.load(open("exp_4_4_data.pkl", "rb"))
    env = ClusterEnv()
    env.state_buffer = data_collector.start_buffer
    env.NUM_CLUSTERS = 10
    env.cluster()
    result_df = proc_results(env.clusters)
    norm = matplotlib.colors.Normalize(vmin=result_df["cluster_id"].min(), vmax=result_df["cluster_id"].max())
    sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
    ax = sns.scatterplot(x=result_df["mobile_pos_x"], y=result_df["mobile_pos_y"], data=result_df, hue="cluster_id", size="total_fuel")
    sm.set_array([])
    ax.figure.colorbar(sm)
    plt.legend([], [], frameon=False)
    ax.set(xlabel="Mobile X-Position", ylabel="Mobile Y-Position")
    plt.title("State Clusters From Stratified Exploration", pad=20)
    plt.savefig("4_4_fig.png")
    plt.show()


if __name__ == "__main__":
    main()

