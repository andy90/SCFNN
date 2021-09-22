# %%
# the main function
from matplotlib.pyplot import axis
import numpy as np
from parameters import *
from rotation_neighbour_dist_functions import *
from produce_features_test import *
from test_networks import *
from useful_functions import *
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
# %%
def get_Oneighbor(Oxyz_all, Hxyz_all, boxlength_all): # get the rOH1 and rOH2 centered around the Oxygen
    Hind_all = ()
    rOH_all = ()
    vecrOH_all = ()
    for ifolder in range(nfolders):
        for iconfig in range(nconfigs_test):
            data_OH = get_neighbour_rij(Oxyz_all[:, :, ifolder, iconfig], Hxyz_all[:, :, ifolder, iconfig], boxlength_all[ifolder, iconfig], 2)
            Hind_all += (data_OH[0], )
            rOH_all += (data_OH[1], )
            vecrOH_all += (data_OH[2], )

    Hind_stack = np.stack(Hind_all, axis=-1).reshape((noxygen, 2, nfolders, nconfigs_test))
    rOH_stack = np.stack(rOH_all, axis=-1).reshape((noxygen, 2, nfolders, nconfigs_test))
    vecrOH_stack = np.stack(vecrOH_all, axis=-1).reshape((noxygen, 2, 3, nfolders, nconfigs_test))
    return Hind_stack, rOH_stack, vecrOH_stack


def get_Hneighbor(Oxyz_all, Hxyz_all, boxlength_all): # get the rOH1 and rOH2 centered around the Oxygen
    Oind_all = ()
    rHO_all = ()
    vecrHO_all = ()

    Hind_all = ()
    rHH_all = ()
    vecrHH_all = ()
    for ifolder in range(nfolders):
        for iconfig in range(nconfigs_test):
            data_HO = get_neighbour_rij(Hxyz_all[:, :, ifolder, iconfig], Oxyz_all[:, :, ifolder, iconfig], boxlength_all[ifolder, iconfig], 1)
            Oind_all += (data_HO[0], )
            rHO_all += (data_HO[1], )
            vecrHO_all += (data_HO[2], )

            data_HH = get_neighbour_rij(Hxyz_all[:, :, ifolder, iconfig], Hxyz_all[:, :, ifolder, iconfig],
                                        boxlength_all[ifolder, iconfig], 2)
            Hind_all += (data_HH[0][:, 1],)
            rHH_all += (data_HH[1][:, 1],)
            vecrHH_all += (data_HH[2][:, 1, :],)

    Oind_stack = np.stack(Oind_all, axis=-1).reshape((nhydrogen, 1, nfolders, nconfigs_test))
    rHO_stack = np.stack(rHO_all, axis=-1).reshape((nhydrogen, 1, nfolders, nconfigs_test))
    vecrHO_stack = np.stack(vecrHO_all, axis=-1).reshape((nhydrogen, 1, 3, nfolders, nconfigs_test))

    Hind_stack = np.stack(Hind_all, axis=-1).reshape((nhydrogen, 1, nfolders, nconfigs_test))
    rHH_stack = np.stack(rHH_all, axis=-1).reshape((nhydrogen, 1, nfolders, nconfigs_test))
    vecrHH_stack = np.stack(vecrHH_all, axis=-1).reshape((nhydrogen, 1, 3, nfolders, nconfigs_test))

    OHind_all = np.concatenate((Oind_stack, Hind_stack), axis=1)
    rHHO_all = np.concatenate((rHO_stack, rHH_stack), axis=1)
    vecrHHO_all = np.concatenate((vecrHO_stack, vecrHH_stack), axis=1)
    return OHind_all, rHHO_all, vecrHHO_all

def generate_rotamers(rOH_nn):
    z_axis = np.array([0, 0, 1])
    y_axis = np.array([0, 1, 0])
    rotamers1 = np.zeros((rOH_nn.shape[0], 3, 3, nfolders, nconfigs_test), dtype=np.float32)
    rotamers2 = np.zeros((rOH_nn.shape[0], 3, 3, nfolders, nconfigs_test), dtype=np.float32)
    for ifolder in range(nfolders):
        for iconfig in range(nconfigs_test):
            for io in range(rOH_nn.shape[0]):
                Ro1 = get_rotation(rOH_nn[io, 0, :, ifolder, iconfig], z_axis)
                rH2Op = Ro1 @ rOH_nn[io, 1, :, ifolder, iconfig]
                Ro2 = get_rotation(np.array([rH2Op[0], rH2Op[1], 0]), y_axis)
                rotamers1[io, :, :, ifolder, iconfig] = Ro1
                rotamers2[io, :, :, ifolder, iconfig] = Ro2
    return rotamers1, rotamers2

def map_wxyz(wxyz_all_rotated): # find out the 4 closest neighbours of the wannier center corresponding to each oxygen
    rij = np.linalg.norm(wxyz_all_rotated, axis=2)
    rij_indmin = np.argsort(rij, axis=1)
    i_indexarray = np.expand_dims(np.arange(rij.shape[0]), axis=(1, 2, 3))
    ifolder_indexarray = np.expand_dims(np.arange(nfolders), axis=(0, 1, 3))
    iconfig_indexarray = np.expand_dims(np.arange(nconfigs_test), axis=(0, 1, 2))

    rij_indmin2 = np.expand_dims(rij_indmin, axis=2)
    i_indexarray2 = np.expand_dims(np.arange(rij.shape[0]), axis=(1, 2, 3, 4))
    ifolder_indexarray2 = np.expand_dims(np.arange(nfolders), axis=(0, 1, 2, 4))
    iconfig_indexarray2 = np.expand_dims(np.arange(nconfigs_test), axis=(0, 1, 2, 3))
    ixyz_indexarray2 = np.expand_dims(np.arange(3), axis=(0, 1, 3, 4))

    rij_min = rij[i_indexarray, rij_indmin, ifolder_indexarray, iconfig_indexarray]
    vecrij_min = wxyz_all_rotated[i_indexarray2, rij_indmin2, ixyz_indexarray2, ifolder_indexarray2, iconfig_indexarray2]

    wannier_xyz_mapped_rotated = vecrij_min[:, 0:4, :, :, :]

    # sort the wannier_xyz_mapped_rotated
    wannier_z_indmin = np.expand_dims(np.argsort(wannier_xyz_mapped_rotated[:, :, 2, :, :], axis=1), axis=2)  # sort according to x axis first
    wannier_xyz_zmin = wannier_xyz_mapped_rotated[i_indexarray2, wannier_z_indmin, ixyz_indexarray2, ifolder_indexarray2, iconfig_indexarray2]

    wannier_xyz_zmin_lower = wannier_xyz_zmin[:, :3, :, :, :] # for the remaining part
    wannier_xz_indmin = np.expand_dims(np.argsort(wannier_xyz_zmin_lower[:, :, 0, :, :], axis=1), axis=2)  # sort according to z axis
    wannier_xyz_xzmin = wannier_xyz_zmin_lower[i_indexarray2, wannier_xz_indmin, ixyz_indexarray2, ifolder_indexarray2, iconfig_indexarray2]

    wannier_xyz_final = np.concatenate((wannier_xyz_xzmin, wannier_xyz_zmin[:, 3:, :, :, :]), axis=1)
    return rij_min[:, 0:4, :, :], wannier_xyz_final

def get_Ewald(Oxyz_all, Hxyz_all, wxyz_all, boxlength_all):
    # calculate the Ewald field from the new wannier center positions. the goal is to reach self consistency
    sigmaE = sigma / np.sqrt(2)  # this sigma is used in literature for Ewald summation

    # calculate the Ewald field
    EO_all = ()
    EH_all = ()
    Ew_all = ()
    for ifolder in range(Oxyz_all.shape[2]):
        for iconfig in range(Oxyz_all.shape[3]):
            boxlength = boxlength_all[ifolder, iconfig]
            Oxyz = Oxyz_all[:, :, ifolder, iconfig]
            Hxyz = Hxyz_all[:, :, ifolder, iconfig]
            wcxyz = wxyz_all[:, :, ifolder, iconfig]

            k0 = 2 * np.pi / boxlength  # the unit of the wavevector
            nkmax = 5
            kxyz = np.zeros(((2 * nkmax - 1)**3-1, 3), dtype=np.float32)  # create the matrix that stores all the wavevectors
            ik = 0
            for nx in range(- nkmax + 1, nkmax):
                for ny in range(- nkmax + 1, nkmax):
                    for nz in range(- nkmax + 1, nkmax):
                        if ((nx != 0) | (ny != 0) | (nz != 0)):
                            kxyz[ik, 0] = nx * k0
                            kxyz[ik, 1] = ny * k0
                            kxyz[ik, 2] = nz * k0
                            ik = ik + 1

            Oxyz_expand = np.expand_dims(Oxyz, axis=1)
            Hxyz_expand = np.expand_dims(Hxyz, axis=1)
            wcxyz_expand = np.expand_dims(wcxyz, axis=1)
            kxyz_expand = np.expand_dims(kxyz, axis=0)

            Sk = np.sum(qO * np.exp(1j * np.sum(Oxyz_expand * kxyz_expand, axis=-1)), axis=0) + np.sum(qH * np.exp(1j * np.sum(Hxyz_expand * kxyz_expand, axis=-1)), axis=0) + np.sum(qw * np.exp(1j * np.sum(wcxyz_expand * kxyz_expand, axis=-1)), axis=0)

            coeff = 2 * np.pi / boxlength ** 3  # the coefficient used before the long ranged contribution

            dSkO = 1j * kxyz_expand * np.expand_dims(np.exp(1j * np.sum(Oxyz_expand * kxyz_expand, axis=-1)), axis=2)  # the derivative of Sk with respect to rO with qO removed
            dSkH = 1j * kxyz_expand * np.expand_dims(np.exp(1j * np.sum(Hxyz_expand * kxyz_expand, axis=-1)), axis=2)  # the derivative of Sk with respect to rH with qH removed
            dSkw = 1j * kxyz_expand * np.expand_dims(np.exp(1j * np.sum(wcxyz_expand * kxyz_expand, axis=-1)), axis=2) # the derivative of Sk with respect to rw with qw removed

            Sk_expand = np.expand_dims(Sk, axis=1)

            knorm = np.linalg.norm(kxyz, axis=-1)
            knorm_expand = np.expand_dims(knorm, axis=-1)

            EO = - coeff * np.sum(np.exp(- (sigmaE * knorm_expand) ** 2 / 2) / knorm_expand ** 2 * 2 * (Sk_expand.conjugate() * dSkO).real, axis=1)
            EH = - coeff * np.sum(np.exp(- (sigmaE * knorm_expand) ** 2 / 2) / knorm_expand ** 2 * 2 * (Sk_expand.conjugate() * dSkH).real, axis=1)
            Ew = - coeff * np.sum(np.exp(- (sigmaE * knorm_expand) ** 2 / 2) / knorm_expand ** 2 * 2 * (Sk_expand.conjugate() * dSkw).real, axis=1)

            EO_all += (EO,)
            EH_all += (EH,)
            Ew_all += (Ew,)

    EO_all_stack = np.stack(EO_all, axis=2)
    EH_all_stack = np.stack(EH_all, axis=2)
    Ew_all_stack = np.stack(Ew_all, axis=2)

    EO_all_reshape = EO_all_stack.reshape((noxygen, 3, nfolders, nconfigs_test))
    EH_all_reshape = EH_all_stack.reshape((nhydrogen, 3, nfolders, nconfigs_test))
    Ew_all_reshape = Ew_all_stack.reshape((nwannier, 3, nfolders, nconfigs_test))

    Eexternal = np.array([0, 0, 0, 0, 0, 0, 0, 0.1/51.4, 0.2/51.4]).reshape((3, nfolders, 1))  # the field is applied to the z-direction, 51.4 is the factor that converts the field from V/A to atomic unit

    EO_sum = EO_all_reshape + Eexternal
    EH_sum = EH_all_reshape + Eexternal
    Ew_sum = Ew_all_reshape + Eexternal

    return EO_sum, EH_sum, Ew_sum


# %%
Oxyz_all = np.load("Oxyz_testconfigs.npy").astype(np.float32)
Hxyz_all = np.load("Hxyz_testconfigs.npy").astype(np.float32)
boxlength_all = np.load("boxlength_testconfigs.npy").astype(np.float32)
wxyz_all = np.load("wxyz_testconfigs.npy").astype(np.float32)
fO_all = np.load("fO_testconfigs.npy").astype(np.float32)
fH_all = np.load("fH_testconfigs.npy").astype(np.float32)
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
#features_wannier_GT_scaled = features_wannier_GT(Oxyz_all_rotated_O[:, :, :, 0, :], Hxyz_all_rotated_O[:, :, :, 0, :])
features_wannier_delta = features_wannier_peturb(Oxyz_all_rotated_O, Hxyz_all_rotated_O, EO_all_rotated_O, EH_all_rotated_O)

# %%
# train for wannier prediction
wannier_rotated_peturb_pred = test_wannier_peturb(wxyz_all_rotated_mapped, features_wannier_delta)
#wxyz_all_rotated_mapped_GT = np.mean(wxyz_all_rotated_mapped - wannier_rotated_peturb_pred, axis=-2)
#wannier_rotated_GT_pred = test_wannier_GT(wxyz_all_rotated_mapped_GT, features_wannier_GT_scaled)
# %% 
wannier_peturb_pred = backrotate_shift(wannier_rotated_peturb_pred, Oxyz_all, R1O, R2O)
wxyz_all_mapped = backrotate_shift(wxyz_all_rotated_mapped, Oxyz_all, R1O, R2O)
# %%
wannier_peturb_to0_pred = wannier_peturb_pred[:, :, :, 1:, :] - np.expand_dims(wannier_peturb_pred[:, :, :, 0, :], axis=-2)
wannier_peturb_to0 = wxyz_all_mapped[:, :, :, 1:, :] - np.expand_dims(wxyz_all_mapped[:, :, :, 0, :], axis=-2)

# %%
print(np.mean(np.abs(wannier_peturb_to0[:, :, 2, 0, :] - wannier_peturb_to0_pred[:, :, 2, 0, :])), np.mean(wannier_peturb_to0[:, :, 2, 0, :]))
print(np.mean(np.abs(wannier_peturb_to0[:, :, 2, 1, :] - wannier_peturb_to0_pred[:, :, 2, 1, :])), np.mean(wannier_peturb_to0[:, :, 2, 1, :]))

print(np.mean(np.abs((wannier_peturb_to0[:, :, 2, 0, :] - wannier_peturb_to0_pred[:, :, 2, 0, :])/wannier_peturb_to0[:, :, 2, 0, :])))
print(np.mean(np.abs((wannier_peturb_to0[:, :, 2, 1, :] - wannier_peturb_to0_pred[:, :, 2, 1, :])/wannier_peturb_to0[:, :, 2, 1, :])))

#np.savetxt("wannier_pred_MSE.txt", np.array([[np.mean(np.abs(wannier_peturb_to0[:, :, 2, 0, :] - wannier_peturb_to0_pred[:, :, 2, 0, :])), np.mean(wannier_peturb_to0[:, :, 2, 0, :])],[np.mean(np.abs(wannier_peturb_to0[:, :, 2, 1, :] - wannier_peturb_to0_pred[:, :, 2, 1, :])), np.mean(wannier_peturb_to0[:, :, 2, 1, :])]])
# %%
wcentroid_peturb_to0_pred = np.mean(wannier_peturb_to0_pred, axis=1)
wcentroid_peturb_to0 = np.mean(wannier_peturb_to0, axis=1)

# %%
print(np.mean(np.abs(wcentroid_peturb_to0[:, 2, 0, :] - wcentroid_peturb_to0_pred[:, 2, 0, :])), np.mean(wcentroid_peturb_to0[:, 2, 0, :]))
print(np.mean(np.abs(wcentroid_peturb_to0[:, 2, 1, :] - wcentroid_peturb_to0_pred[:, 2, 1, :])), np.mean(wcentroid_peturb_to0[:, 2, 1, :]))

print(np.mean(np.abs((wcentroid_peturb_to0[:, 2, 0, :] - wcentroid_peturb_to0_pred[:, 2, 0, :])/wcentroid_peturb_to0[:, 2, 0, :])))
print(np.mean(np.abs((wcentroid_peturb_to0[:, 2, 1, :] - wcentroid_peturb_to0_pred[:, 2, 1, :])/wcentroid_peturb_to0[:, 2, 1, :])))

np.savetxt("wannier_centroid_pred_MSE_MAPE.txt", np.array([[np.mean(np.abs(wcentroid_peturb_to0[:, 2, 0, :] - wcentroid_peturb_to0_pred[:, 2, 0, :])), np.mean(wcentroid_peturb_to0[:, 2, 0, :]), np.mean(np.abs((wcentroid_peturb_to0[:, 2, 0, :] - wcentroid_peturb_to0_pred[:, 2, 0, :])/wcentroid_peturb_to0[:, 2, 0, :]))],[np.mean(np.abs(wcentroid_peturb_to0[:, 2, 1, :] - wcentroid_peturb_to0_pred[:, 2, 1, :])), np.mean(wcentroid_peturb_to0[:, 2, 1, :]), np.mean(np.abs((wcentroid_peturb_to0[:, 2, 1, :] - wcentroid_peturb_to0_pred[:, 2, 1, :])/wcentroid_peturb_to0[:, 2, 1, :]))]]))
# %%
#print(np.mean(np.abs(wannier_peturb_to0[:, :, 2, 0, :] - wannier_peturb_to0_pred[:, :, 2, 0, :])), np.mean(wannier_peturb_to0[:, :, 2, 0, :]))
#print(np.mean(np.abs(wcentroid_peturb_to0[:, 2, 1, :] - wcentroid_peturb_to0_pred[:, 2, 1, :])), np.mean(wcentroid_peturb_to0[:, 2, 1, :]))

# %%
#wc_1z_hist = np.histogram(wcentroid_peturb_to0[:, 2, 0, :], range=(-0.005,0.), bins=200)
#wc_1z_hist_pred = np.histogram(wcentroid_peturb_to0_pred[:, 2, 0, :], range=(-0.005,0.), bins=200)

#wc_2z_hist = np.histogram(wcentroid_peturb_to0[:, 2, 1, :], range=(-0.01,0.), bins=200)
#wc_2z_hist_pred = np.histogram(wcentroid_peturb_to0_pred[:, 2, 1, :], range=(-0.01,0.), bins=200)
# %%
#plt.plot(wc_1z_hist[1][1:], wc_1z_hist[0], 'o')
#plt.plot(wc_1z_hist_pred[1][1:], wc_1z_hist_pred[0], '-')

#plt.plot(wc_2z_hist[1][1:], wc_2z_hist[0], 'o')
#plt.plot(wc_2z_hist_pred[1][1:], wc_2z_hist_pred[0], '-')

# %%
#wc_1r_hist = np.histogram(np.linalg.norm(wcentroid_peturb_to0[:, :, 0, :], axis=1), range=(0,0.005), bins=200)
#wc_1r_hist_pred = np.histogram(np.linalg.norm(wcentroid_peturb_to0_pred[:, :, 0, :], axis=1), range=(0,0.005), bins=200)

#wc_2z_hist = np.histogram(wcentroid_peturb_to0[:, 2, 1, :], range=(-0.01,0.), bins=200)
#wc_2z_hist_pred = np.histogram(wcentroid_peturb_to0_pred[:, 2, 1, :], range=(-0.01,0.), bins=200)
# %%
#plt.plot(wc_1r_hist[1][1:], wc_1r_hist[0], 'o')
#plt.plot(wc_1r_hist_pred[1][1:], wc_1r_hist_pred[0], '-')

# %%
#wcentroid_peturb_to0_pred_total = np.mean(wannier_peturb_to0_pred, axis=(0, 1))
#wcentroid_peturb_to0_total = np.mean(wannier_peturb_to0, axis=(0,1))

# %%
#wc_1z_hist_total = np.histogram(wcentroid_peturb_to0_total[2, 0], range=(-0.003,-0.002), bins=20)
#wc_1z_hist_pred_total = np.histogram(wcentroid_peturb_to0_pred_total[2, 0], range=(-0.003,-0.002), bins=20)
#plt.plot(wc_1z_hist_total[1][1:], wc_1z_hist_total[0], 'o')
#plt.plot(wc_1z_hist_pred_total[1][1:], wc_1z_hist_pred_total[0], '-')

# %%
p2_OO = np.loadtxt("fG2_parameters_OO.txt")
p2_OH = np.loadtxt("fG2_parameters_OH.txt")
p4_OO = np.loadtxt("fG4_parameters_OO.txt")
p4_OH = np.loadtxt("fG4_parameters_OH.txt")
features_force_peturb_O = features_force_peturb(Oxyz_all_rotated_O, Hxyz_all_rotated_O, EO_all_rotated_O, EH_all_rotated_O, p2_OO, p2_OH, p4_OO, p4_OH)
    
# %%
# train for the force fO, reuse the features for wannier peturb prediction
fO_rotated_peturb_pred = test_force_peturb(fO_all_rotated, features_force_peturb_O, "O")

# %%
fO_peturb_pred = backrotate(fO_rotated_peturb_pred, R1O, R2O)
fO_peturb_to0_pred = fO_peturb_pred[:, :, 1:, :] - np.expand_dims(fO_peturb_pred[:, :, 0, :], axis=-2)
fO_peturb_to0 = fO_all[:, :, 1:, :] - np.expand_dims(fO_all[:, :, 0, :], axis=-2)

# %%
print(np.mean(np.abs(fO_peturb_to0_pred[:, 2, 0, :] - fO_peturb_to0[:, 2, 0, :])), np.mean(fO_peturb_to0[:, 2, 0, :]))
print(np.mean(np.abs(fO_peturb_to0_pred[:, 2, 1, :] - fO_peturb_to0[:, 2, 1, :])), np.mean(fO_peturb_to0[:, 2, 1, :]))

print(np.mean(np.abs((fO_peturb_to0_pred[:, 2, 0, :] - fO_peturb_to0[:, 2, 0, :])/fO_peturb_to0[:, 2, 0, :])))
print(np.mean(np.abs((fO_peturb_to0_pred[:, 2, 1, :] - fO_peturb_to0[:, 2, 1, :])/fO_peturb_to0[:, 2, 1, :])))

np.savetxt("fO_pred_MSE_MAPE.txt", np.array([[np.mean(np.abs(fO_peturb_to0_pred[:, 2, 0, :] - fO_peturb_to0[:, 2, 0, :])), np.mean(fO_peturb_to0[:, 2, 0, :]), np.mean(np.abs((fO_peturb_to0_pred[:, 2, 0, :] - fO_peturb_to0[:, 2, 0, :])/fO_peturb_to0[:, 2, 0, :]))],[np.mean(np.abs(fO_peturb_to0_pred[:, 2, 1, :] - fO_peturb_to0[:, 2, 1, :])), np.mean(fO_peturb_to0[:, 2, 1, :]), np.mean(np.abs((fO_peturb_to0_pred[:, 2, 1, :] - fO_peturb_to0[:, 2, 1, :])/fO_peturb_to0[:, 2, 1, :]))]]))
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
features_force_peturb_H = features_force_peturb(Hxyz_all_rotated_H, Oxyz_all_rotated_H, EH_all_rotated_H, EO_all_rotated_H, p2_HH, p2_HO, p4_HH, p4_HO)

# %%
fH_rotated_peturb_pred = test_force_peturb(fH_all_rotated, features_force_peturb_H, "H")

# %%
fH_peturb_pred = backrotate(fH_rotated_peturb_pred, R1H, R2H)
fH_peturb_to0_pred = fH_peturb_pred[:, :, 1:, :] - np.expand_dims(fH_peturb_pred[:, :, 0, :], axis=-2)
fH_peturb_to0 = fH_all[:, :, 1:, :] - np.expand_dims(fH_all[:, :, 0, :], axis=-2)

# %%
print(np.mean(np.abs(fH_peturb_to0_pred[:, 2, 0, :] - fH_peturb_to0[:, 2, 0, :])), np.mean(fH_peturb_to0[:, 2, 0, :]))
print(np.mean(np.abs(fH_peturb_to0_pred[:, 2, 1, :] - fH_peturb_to0[:, 2, 1, :])), np.mean(fH_peturb_to0[:, 2, 1, :]))

print(np.mean(np.abs((fH_peturb_to0_pred[:, 2, 0, :] - fH_peturb_to0[:, 2, 0, :])/fH_peturb_to0[:, 2, 0, :])))
print(np.mean(np.abs((fH_peturb_to0_pred[:, 2, 1, :] - fH_peturb_to0[:, 2, 1, :])/fH_peturb_to0[:, 2, 1, :])))

np.savetxt("fH_pred_MSE_MAPE.txt", np.array([[np.mean(np.abs(fH_peturb_to0_pred[:, 2, 0, :] - fH_peturb_to0[:, 2, 0, :])), np.mean(fH_peturb_to0[:, 2, 0, :]), np.mean(np.abs((fH_peturb_to0_pred[:, 2, 0, :] - fH_peturb_to0[:, 2, 0, :])/fH_peturb_to0[:, 2, 0, :]))],[np.mean(np.abs(fH_peturb_to0_pred[:, 2, 1, :] - fH_peturb_to0[:, 2, 1, :])), np.mean(fH_peturb_to0[:, 2, 1, :]), np.mean(np.abs((fH_peturb_to0_pred[:, 2, 1, :] - fH_peturb_to0[:, 2, 1, :])/fH_peturb_to0[:, 2, 1, :]))]]))
# %%
#fO_GT = np.mean(fO_all - fO_peturb_pred, axis=-2)
#fH_GT = np.mean(fH_all - fH_peturb_pred, axis=-2)
# %%
#fO_GT_pred, fH_GT_pred =  train_force_GT(fO_GT, fH_GT)
# %%
