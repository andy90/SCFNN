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
# this flag controls whether we need to produce features
pf = True

# %%
Oxyz_all = np.load("Oxyz_allconfigs.npy").astype(np.float32)
Hxyz_all = np.load("Hxyz_allconfigs.npy").astype(np.float32)
boxlength_all = np.load("boxlength_allconfigs.npy").astype(np.float32)
wxyz_all = np.load("wxyz_allconfigs.npy").astype(np.float32)
fO_all = np.load("fO_allconfigs.npy").astype(np.float32)
fH_all = np.load("fH_allconfigs.npy").astype(np.float32)
neighbour_O = get_Oneighbor(Oxyz_all, Hxyz_all, boxlength_all)
neighbour_H = get_Hneighbor(Oxyz_all, Hxyz_all, boxlength_all)

EO_all, EH_all, Ew_all = get_Ewald(Oxyz_all, Hxyz_all, wxyz_all, boxlength_all)


# %%
R1O, R2O = generate_rotamers(neighbour_O[2])
EO_all_rotated_O = rotate(EO_all, R1O, R2O)
EH_all_rotated_O = rotate(EH_all, R1O, R2O)
Oxyz_all_rotated_O = shift_rotate(Oxyz_all, Oxyz_all, boxlength_all, R1O, R2O)
Hxyz_all_rotated_O = shift_rotate(Hxyz_all, Oxyz_all, boxlength_all, R1O, R2O)
wxyz_all_rotated_O = shift_rotate(wxyz_all, Oxyz_all, boxlength_all, R1O, R2O)
index_O = np.arange(noxygen)
fO_all_rotated_all = rotate(fO_all, R1O, R2O)
fO_all_rotated = fO_all_rotated_all[index_O, index_O, :, :, :]
rwxyz_mapped, wxyz_all_rotated_mapped = map_wxyz(wxyz_all_rotated_O)

# %%
# produce features
if pf:
    features_wannier_GT_scaled = features_wannier_GT(Oxyz_all_rotated_O[:, :, :, 0, :], Hxyz_all_rotated_O[:, :, :, 0, :])
    features_wannier_delta = features_wannier_peturb(Oxyz_all_rotated_O, Hxyz_all_rotated_O, EO_all_rotated_O, EH_all_rotated_O)
    np.save("features_wannier_GT_scaled", features_wannier_GT_scaled)
    np.save("features_wannier_peturb", features_wannier_delta)

# %%
# train for wannier prediction
features_wannier_delta = np.load("features_wannier_peturb.npy")
features_wannier_GT_scaled = np.load("features_wannier_GT_scaled.npy")
wannier_rotated_peturb_pred = train_wannier_peturb(wxyz_all_rotated_mapped, features_wannier_delta)
wxyz_all_rotated_mapped_GT = np.mean(wxyz_all_rotated_mapped - wannier_rotated_peturb_pred, axis=-2)
wannier_rotated_GT_pred = train_wannier_GT(wxyz_all_rotated_mapped_GT, features_wannier_GT_scaled)

# %%
if pf:
    p2_OO = np.loadtxt("fG2_parameters_OO.txt")
    p2_OH = np.loadtxt("fG2_parameters_OH.txt")
    p4_OO = np.loadtxt("fG4_parameters_OO.txt")
    p4_OH = np.loadtxt("fG4_parameters_OH.txt")
    features_force_peturb_O = features_force_peturb(Oxyz_all_rotated_O, Hxyz_all_rotated_O, EO_all_rotated_O, EH_all_rotated_O, p2_OO, p2_OH, p4_OO, p4_OH)
    np.save("features_force_peturb_O", features_force_peturb_O)
    
# %%
# train for the force fO, reuse the features for wannier peturb prediction
features_force_peturb_O = np.load("features_force_peturb_O.npy")
fO_rotated_peturb_pred = train_force_peturb(fO_all_rotated, features_force_peturb_O, "O")

# %%
# rotate around H and predict the force on H
R1H, R2H = generate_rotamers(neighbour_H[2])
EO_all_rotated_H = rotate(EO_all, R1H, R2H)
EH_all_rotated_H = rotate(EH_all, R1H, R2H)
Oxyz_all_rotated_H = shift_rotate(Oxyz_all, Hxyz_all, boxlength_all, R1H, R2H)
Hxyz_all_rotated_H = shift_rotate(Hxyz_all, Hxyz_all, boxlength_all, R1H, R2H)
index_H = np.arange(nhydrogen)
fH_all_rotated_all = rotate(fH_all, R1H, R2H)
fH_all_rotated = fH_all_rotated_all[index_H, index_H, :, :, :]
p2_HH = np.loadtxt("fG2_parameters_HH.txt")
p2_HO = np.loadtxt("fG2_parameters_HO.txt")
p4_HH = np.loadtxt("fG4_parameters_HH.txt")
p4_HO = np.loadtxt("fG4_parameters_HO.txt")

# %%
if pf:
    features_force_peturb_H = features_force_peturb(Hxyz_all_rotated_H, Oxyz_all_rotated_H, EH_all_rotated_H, EO_all_rotated_H, p2_HH, p2_HO, p4_HH, p4_HO)
    np.save("features_force_peturb_H", features_force_peturb_H)

# %%
features_force_peturb_H = np.load("features_force_peturb_H.npy")
fH_rotated_peturb_pred = train_force_peturb(fH_all_rotated, features_force_peturb_H, "H")

# %%
fH_peturb_pred = backrotate(fH_rotated_peturb_pred, R1H, R2H)
fO_peturb_pred = backrotate(fO_rotated_peturb_pred, R1O, R2O)

# %%
fO_GT = np.mean(fO_all - fO_peturb_pred, axis=-2)
fH_GT = np.mean(fH_all - fH_peturb_pred, axis=-2)
# %%
fO_GT_pred, fH_GT_pred =  train_force_GT(fO_GT, fH_GT)
# %%
