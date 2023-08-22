import pickle

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

with open('exp_2_1_data.pkl', 'rb') as f:
    data_dict = pickle.load(f)

heatplot_array = np.zeros((4, 5))
for i, bound in enumerate([1e4, 1e5, 1e6, 1e7]):
    for j, thrust in enumerate([1, 5, 10, 50, 100]):
        one_dict = data_dict[(thrust, bound)]
        capture_prop = round(one_dict['num_captures'] /
                             one_dict['episodes'], 3)
        heatplot_array[i, j] = capture_prop

x = sns.color_palette("mako", as_cmap=True)
hm = sns.heatmap(heatplot_array, annot=True, fmt='g', cmap=x,
                 cbar_kws={'label': 'Capture Proportion'})
hm.set_xticklabels([1, 5, 10, 50, 100])
hm.set_yticklabels(['1e4', '1e5', '1e6', '1e7'])
hm.set_ylabel('Capture Radius')
hm.set_xlabel('Max Thrust')
plt.gca().invert_yaxis()
plt.show()
