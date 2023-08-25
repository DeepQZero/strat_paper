import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import tensorboard as tb

def load_experiment():
    exp_id = ""
    experiment = tb.data.experimental.ExperimentFromDev(exp_id)
    exp_df = experiment.get_scalars()
    return exp_df

def main():
    exp_df = load_experiment()
    print(exp_df)
    return

if __name__ == "__main__":
    main()