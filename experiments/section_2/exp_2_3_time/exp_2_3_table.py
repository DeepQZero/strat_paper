import pickle

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

unpickled_dict = None
with open('exp_2_3_data.pkl','rb') as f:
    unpickled_dict = pickle.load(f)

print(unpickled_dict)

raw_map = np.zeros((5, 7))

for i, UPPER_THRUST in enumerate([10, 5, 2, np.sqrt(2), 1]):
    small_dict = unpickled_dict[UPPER_THRUST]
    for j, ANGLE_DIFF in enumerate([round(np.pi/8 * i, 2) for i in range(1, 8)]):
        raw_map[i, j] = small_dict[ANGLE_DIFF]['third']

hm = sns.heatmap(raw_map, annot=True)
plt.show()