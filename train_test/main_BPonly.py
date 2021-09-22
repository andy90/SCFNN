# %%
# the main function
import numpy as np
from parameters import *
from get_Ewald import *
from rotation_neighbour_dist_functions import *
from produce_features import *
from train_networks import *
from useful_functions import *


# %%
Oxyz_all = np.load("Oxyz_allconfigs.npy").astype(np.float32)
Hxyz_all = np.load("Hxyz_allconfigs.npy").astype(np.float32)
boxlength_all = np.load("boxlength_allconfigs.npy").astype(np.float32)
fO_all = np.load("fO_allconfigs.npy").astype(np.float32)
fH_all = np.load("fH_allconfigs.npy").astype(np.float32)
# %%

# %%
fO_pred, fH_pred =  train_force_BP(fO_all[:, :, 0, :], fH_all[:, :, 0, :])
# %%
