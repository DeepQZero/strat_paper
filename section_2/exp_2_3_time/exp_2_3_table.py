import pickle

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

with open('exp_2_3_data.pkl', 'rb') as f:
    data_dict = pickle.load(f)

heatplot_array = np.zeros((7, 5))
for i, zone in enumerate([round(np.pi/8 * i, 2) for i in range(1, 8)]):
    for j, thrust in enumerate([5, 7.5, 10, 15, 20]):
        heatplot_array[i, j] = data_dict[thrust][zone]['num_entrances'] / 1e5

x = sns.color_palette("mako", as_cmap=True)
hm = sns.heatmap(heatplot_array, annot=True, fmt='g', cmap=x,
                 cbar_kws={'label': 'Average First Time Step in Zone'})
hm.set_xticklabels([5, 7.5, 10, 15, 20])
hm.set_yticklabels([round(np.pi/8 * i, 2) for i in range(1, 8)])
hm.set_ylabel('Zone Upper Bound')
hm.set_xlabel('Max Thrust')
plt.gca().invert_yaxis()
plt.show()
