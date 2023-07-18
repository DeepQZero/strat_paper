import pickle

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

unpickled_dict = None
with open('exp_2_2_data.pkl','rb') as f:
    unpickled_dict = pickle.load(f)

print(unpickled_dict)

raw_map = np.zeros((8, 5))

for r, PASSIVE_PROP in enumerate([0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]):
    for c, UPPER_THRUST in enumerate([1, np.sqrt(2), 2, 5, 10]):
        small_dict = unpickled_dict[(PASSIVE_PROP, UPPER_THRUST)]
        capture_ratio = round(small_dict['captures'] / small_dict['times'], 2)
        raw_map[r, c] = capture_ratio
        # print(PASSIVE_PROP, UPPER_THRUST, capture_ratio)

x = sns.color_palette("mako", as_cmap=True)
hm = sns.heatmap(raw_map, annot=True, fmt='g', cmap=x, cbar_kws={'label': 'Capture Prop'})
hm.set_xticklabels([1, 1.41, 2, 5, 10])
hm.set_yticklabels([0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0])
hm.set_ylabel('Passive Proportion')
hm.set_xlabel('Max Thrust')
plt.gca().invert_yaxis()
plt.show()