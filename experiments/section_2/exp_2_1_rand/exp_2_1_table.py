import pickle

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

unpickled_dict = None
with open('exp_2_1_data.pkl','rb') as f:
    unpickled_dict = pickle.load(f)

print(unpickled_dict)

raw_map = np.zeros((4, 5))

for i, BOUND in enumerate([1e4, 1e5, 1e6, 1e7]):
    for j, UPPER_THRUST in enumerate([0.5, 1, 2, 5, 10]):
        small_dict = unpickled_dict[(UPPER_THRUST, BOUND)]
        capture_ratio = round(small_dict['captures'] / small_dict['times'], 2)
        raw_map[i, j] = capture_ratio
        # print(UPPER_THRUST, BOUND, capture_ratio)

x = sns.color_palette("mako", as_cmap=True)
hm = sns.heatmap(raw_map, annot=True, fmt='g', cmap=x, cbar_kws={'label': 'Capture Prop'})
hm.set_xticklabels([0.5, 1, 2, 5, 10])
hm.set_yticklabels(['1e4', '1e5', '1e6', '1e7'])
hm.set_ylabel('Capture Radius')
hm.set_xlabel('Max Thrust')
plt.gca().invert_yaxis()
plt.show()