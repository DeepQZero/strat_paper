import pickle

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# with open('../../section_2/exp_2_3_time/exp_2_3_data.pkl', 'rb') as f:
#     data_dict_fixed = pickle.load(f)

with open('exp_3_2_data.pkl', 'rb') as f:
     data_dict_fixed = pickle.load(f)

with open('exp_3_1_data.pkl', 'rb') as f:
    data_dict_random = pickle.load(f)

# tmp 1 is -np.pi/8 cont
# tmp 2 is -np.pi/4 cont
# tmp 3 is -np.pi/4 discrete 3
# tmp 4 is with bound, fixed
# tmp 5 is -np.pi/8 cont
# tmp 6 is -np.pi/4 cont
# tmp 7 is -np.pi/4 disc
# tmp 8 is fuel env with bounds, fixed
# tmp 9 is fuel env with bounds, fixed at one state np.pi/4
# tmp 10 is fuel env with bounds, fixed at one state np.pi/8
# tmp 11 is fuel env with bounds, cont -np.pi/8, np.pi/8
# tmp 12 retry of tmp 11

heatplot_array = np.zeros((7, 5))
for i, zone in enumerate([round(np.pi/8 * i, 2) for i in range(1, 8)]):
    for j, thrust in enumerate([5, 7.5, 10, 15, 20]):
        heatplot_array[i, j] = data_dict_fixed[thrust][zone]['num_entrances'] / 1e4 - \
                               data_dict_random[thrust][zone]['num_entrances'] / 1e4

x = sns.color_palette("mako", as_cmap=True)
hm = sns.heatmap(heatplot_array, annot=True, fmt='g', cmap=x,
                 cbar_kws={'label': 'Caption'})
hm.set_xticklabels([5, 7.5, 10, 15, 20])
hm.set_yticklabels([round(np.pi/8 * i, 2) for i in range(1, 8)])
hm.set_ylabel('Zone Upper Bound')
hm.set_xlabel('Max Thrust')
plt.gca().invert_yaxis()
plt.show()

