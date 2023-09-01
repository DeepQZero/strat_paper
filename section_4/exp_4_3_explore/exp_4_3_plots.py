import pickle

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
palette = sns.cubehelix_palette(reverse=True, as_cmap=True)

from lib import dynamics as dyn

def proc_results(result_hist):
    result_df = pd.DataFrame(columns=["mobile_pos_x", "mobile_pos_y", "mobile_vel", "capture_pos", "capture_vel",
                                      "return_pos", "return_vel", "turn_num", "total_fuel"])
    for result in result_hist:
        mobile_pos = result[0:2]
        mobile_vel = result[2:4]
        capture_pos = result[4:6]
        capture_vel = result[6:8]
        return_pos = result[8:10]
        return_vel = result[10:12]
        turn_num = (112 - result[12]) * 100/112
        total_fuel = (125 - result[13])*100/125  # Remaining fuel, not fuel used. For color grading

        # Rotate the frame so the capture position is always at zero
        BASE_STATE = np.array([-dyn.GEO, 0.0])
        angle = float(dyn.abs_angle_diff(capture_pos, BASE_STATE))
        mobile_pos = dyn.vec_rotate(mobile_pos, angle)
        capture_pos = dyn.vec_rotate(capture_pos, angle)

        result_dict = {"mobile_pos_x": mobile_pos[0], "mobile_pos_y": mobile_pos[1], "mobile_vel": mobile_vel,
                       "capture_pos": capture_pos, "capture_vel": capture_vel,
                       "return_pos": return_pos, "return_vel": return_vel,
                       "turn_num": turn_num, "total_fuel": total_fuel}
        result_df.loc[len(result_df.index)] = result_dict
    return result_df

def main():
    result_hist = pickle.load(open("exp_4_3_data.pkl", "rb"))
    result_df = proc_results(result_hist)
    norm = matplotlib.colors.Normalize(vmin=result_df["total_fuel"].min(), vmax=result_df["total_fuel"].max())
    sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
    ax = sns.scatterplot(x=result_df["mobile_pos_x"], y=result_df["mobile_pos_y"], data=result_df, hue="total_fuel", size="turn_num")
    sm.set_array([])
    ax.figure.colorbar(sm)
    plt.legend([], [], frameon=False)
    ax.set(xlabel="Mobile X-Position", ylabel="Mobile Y-Position")
    plt.title("Promising States From Directed Exploration", pad=20)
    plt.savefig("4_3_fig.png")
    plt.show()


if __name__ == "__main__":
    main()