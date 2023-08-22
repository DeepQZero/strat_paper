import pickle

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

with open('exp_2_1_data.pkl', 'rb') as f:
    data_dict = pickle.load(f)

heatplot_array = np.zeros((5, 5))
for row, radius in enumerate([1e5, 5e5, 1e6, 5e6, 1e7]):
    for column, thrust in enumerate([5, 7.5, 10, 15, 20]):
        one_dict = data_dict[(thrust, radius)]
        capture_prop = round(one_dict['num_rendez'] /
                             one_dict['episodes'], 3)
        heatplot_array[row, column] = capture_prop

x = sns.color_palette("mako", as_cmap=True)
hm = sns.heatmap(heatplot_array, annot=True, fmt='g', cmap=x,
                 cbar_kws={'label': 'Rendezvous Proportion'})
hm.set_xticklabels([5, 7.5, 10, 15, 20])
hm.set_yticklabels(['1e5', '5e5', '1e6', '5e6', '1e7'])
hm.set_ylabel('Rendezvous Radius')
hm.set_xlabel('Max Thrust')
plt.gca().invert_yaxis()
plt.show()
