import numpy as np

noxygen = 64
natoms = 192
nhydrogen = natoms - noxygen

nfolders = 3
nwannier = noxygen * 4
folder_names = ["D0", "D0p1V", "D0p2V"]

sigma = 8  # the smoothing length sigma for GT cutoff
qO = 6  # charge on the oxygen
qH = 1  # charge on the hydrogen
qw = -2  # charge on the wannier center

bad_configs = np.loadtxt("bad_configurations.txt", dtype=int)
all_configs = np.arange(1, 1000+1)
good_configs = []
for i in all_configs:
    if np.sum(i == bad_configs) == 0 :
        good_configs.append(i)

nconfigs = len(good_configs)

test_configs = []
for i in np.arange(1001, 1594):
    if np.sum(i == bad_configs) == 0 :
        test_configs.append(i)

nconfigs_test = len(test_configs)