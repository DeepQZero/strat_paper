import pickle

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

unpickled_dict = None
with open('exp_2_1_data.pkl','rb') as f:
    unpickled_dict = pickle.load(f)

print(unpickled_dict)

raw_map = np.zeros((5, 5))

for i, UPPER_THRUST in enumerate([10, 5, 2, np.sqrt(2), 1]):
    for j, BOUND in enumerate([1e7, 1e6, 1e5, 1e4, 1e3]):
        small_dict = unpickled_dict[(BOUND, UPPER_THRUST)]
        raw_map[i, j] = round(small_dict['captures'] / small_dict['times'], 2)

hm = sns.heatmap(raw_map, annot=True)
plt.show()