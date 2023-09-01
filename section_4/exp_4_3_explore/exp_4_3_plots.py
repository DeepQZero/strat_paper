import pickle

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
        turn_num = result[12]
        total_fuel = 125 - result[13]  # Remaining fuel, not fuel used for color grading
        result_dict = {"mobile_pos_x": mobile_pos[0], "mobile_pos_y": mobile_pos[1], "mobile_vel": mobile_vel,
                       "capture_pos": capture_pos, "capture_vel": capture_vel,
                       "return_pos": return_pos, "return_vel": return_vel,
                       "turn_num": turn_num, "total_fuel": total_fuel}
        result_df.loc[len(result_df.index)] = result_dict
    return result_df

def main():
    result_hist = pickle.load(open("exp_4_3_data.pkl", "rb"))
    result_df = proc_results(result_hist)
    sns.scatterplot(x=result_df["mobile_pos_x"], y=result_df["mobile_pos_y"])
    plt.savefig("4_3_fig.png")
    plt.show()


if __name__ == "__main__":
    main()