import pickle

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

unpickled_dict = None
with open('../../section_2/exp_2_3_time/exp_2_3_data.pkl','rb') as f:
    unpickled_dict = pickle.load(f)

unpickled_dict2 = None
with open('exp_3_1_data.pkl','rb') as f:
    unpickled_dict2 = pickle.load(f)

print(unpickled_dict)

raw_map = np.zeros((7, 5))

for i, UPPER_THRUST in enumerate([1, np.sqrt(2), 2, 5, 10]):
    small_dict = unpickled_dict[UPPER_THRUST]
    small_dict2 = unpickled_dict2[UPPER_THRUST]
    for j, ANGLE_DIFF in enumerate([round(np.pi/8 * i, 2) for i in range(1, 8)]):
        raw_map[j, i] = round(small_dict2[ANGLE_DIFF]['first'], 1) - round(small_dict[ANGLE_DIFF]['first'], 1)

colormap = sns.color_palette("Blues", as_cmap=True)
x = sns.color_palette("mako", as_cmap=True)
hm = sns.heatmap(raw_map, annot=True, fmt='g', cmap=x, cbar_kws={'label': 'Avg Time'})
hm.set_xticklabels([1, 1.41, 2, 5, 10])
hm.set_yticklabels([round(np.pi/8 * i, 2) for i in range(1, 8)])
hm.set_ylabel('Zone Angle')
hm.set_xlabel('Max Thrust')
hm.set_title('Title')
plt.gca().invert_yaxis()
plt.show()
