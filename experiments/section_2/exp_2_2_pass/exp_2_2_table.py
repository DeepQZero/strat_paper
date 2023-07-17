import pickle

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

unpickled_dict = None
with open('exp_2_2_data.pkl','rb') as f:
    unpickled_dict = pickle.load(f)

print(unpickled_dict)

raw_map = np.zeros((5, 8))

for i, UPPER_THRUST in enumerate([10, 5, 2, np.sqrt(2), 1]):
    for j, PASSIVE_PROP in enumerate([0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]):
        small_dict = unpickled_dict[(PASSIVE_PROP, UPPER_THRUST)]
        raw_map[i, j] = round(small_dict['captures'] / small_dict['times'], 2)

hm = sns.heatmap(raw_map, annot=True)
plt.show()