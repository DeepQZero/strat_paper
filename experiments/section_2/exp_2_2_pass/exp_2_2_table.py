import pickle

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

with open('exp_2_2_data.pkl', 'rb') as f:
    data_dict = pickle.load(f)

heatplot_array = np.zeros((7, 5))
for i, passive_prop in enumerate([0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]):
    for j, thrust in enumerate([0.5, 1, 2, 5, 10]):
        one_dict = data_dict[(thrust, passive_prop)]
        capture_prop = round(one_dict['num_captures'] /
                             one_dict['episodes'], 3)
        heatplot_array[i, j] = capture_prop

x = sns.color_palette("mako", as_cmap=True)
hm = sns.heatmap(heatplot_array, annot=True, fmt='g', cmap=x,
                 cbar_kws={'label': 'Capture Proportion'})
hm.set_xticklabels([0.5, 1, 2, 5, 10])
hm.set_yticklabels([0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
hm.set_ylabel('Passive Action Proportion')
hm.set_xlabel('Max Thrust')
# plt.gca().invert_yaxis()
plt.show()

