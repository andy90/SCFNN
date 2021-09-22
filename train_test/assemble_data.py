import numpy as np
from parameters import *

wxyz_all = ()  # the tuple that stores all the wannier xyz
Oxyz_all = ()  # the tuple that stores all the Oxygen xyz
Hxyz_all = ()  # the tuple that stores all the Hydrogen xyz
fO_all = ()
fH_all = ()
boxlength_all = ()  # the tuple that stores all the boxlengths



for ifolder in folder_names:
    for iconfig in good_configs:  # 801 is the number of Config folders we have
        print(ifolder, iconfig)
        fxyz_name = ifolder + "/Config" + str(iconfig) + "/W64-bulk-HOMO_centers_s1-1_0.xyz"  # this files stores the nucleus and wannier xyz
        fxyz = open(fxyz_name, "r")

        lines_all = fxyz.readlines()  # read in all the lines for the fxyz file
        nxyz_lines = int(lines_all[0].split()[0])  # the total number of lines that stores xyz data of nuclues and wannier center

        # read in the coordinates of the nucleus
        lines_nucleus_start_index = 2  # start from the third line
        nucleus_xyz = np.zeros((natoms, 3))
        for i in range(lines_nucleus_start_index, (lines_nucleus_start_index+natoms)):
            nucleus_xyz[i-lines_nucleus_start_index, :] = np.array(lines_all[i].split()[1:4], dtype=np.float64)

        nucleus_xyz = nucleus_xyz / 0.529177  # change the unit from Angstrom to Bohr

        # read in the coordinates of the wannier centers
        wannier_xyz = np.zeros((nwannier, 3))

        lines_wannier_start_index = lines_nucleus_start_index + natoms
        lines_wannier_end_index = lines_wannier_start_index + nwannier

        for i in range(lines_wannier_start_index, lines_wannier_end_index):
            wannier_xyz[i-lines_wannier_start_index, :] = np.array(lines_all[i].split()[1:4], dtype=np.float64)

        wannier_xyz = wannier_xyz / 0.529177

        # implement the periodic boundary condition, centered around 0
        finit_name = ifolder + "/Config" + str(iconfig) + "/init.xyz"  # read in the box length from the init file
        finit = open(finit_name, "r")
        finit_lines = finit.readlines()
        boxlength = float(finit_lines[1].split()[10]) # box length is already in unit of Bohr

        nucleus_xyz = nucleus_xyz - np.round(nucleus_xyz / boxlength) * boxlength  # enforce pbc, the center of the box is 0
        wannier_xyz = wannier_xyz - np.round(wannier_xyz / boxlength) * boxlength

        # extract the coordinates of the oxygen and hydrogen
        Oxygen_xyz = nucleus_xyz[::3, :]  # subslicing nucleus_xyz, whose first dim is the index for the atoms
        index_H = np.sort(np.concatenate((np.arange(1, 191, 3), np.arange(2, 192, 3))))
        H_xyz = nucleus_xyz[index_H, :]  # subindex nucleus_xyz to get the Hydrogen xyz

        # read in the forces from files
        fforce_name = ifolder + "/Config" + str(iconfig) + "/W64-bulk-W64-forces-1_0.xyz"
        fforce = open(fforce_name, "r")

        lines_all = fforce.readlines()

        fforce_total = np.zeros((natoms, 3))
        for i in range(4, 4 + natoms):
            fforce_total[i - 4, :] = np.array(lines_all[i].split()[3:6], dtype=np.float64)

        fO_total = fforce_total[::3, :]
        fH_total = fforce_total[index_H, :]

        # assemble data
        Oxyz_all += (Oxygen_xyz, )
        Hxyz_all += (H_xyz, )
        wxyz_all += (wannier_xyz, )
        boxlength_all += (boxlength, )
        fO_all += (fO_total,)
        fH_all += (fH_total,)

        np.savetxt(ifolder + "/Config" + str(iconfig) + "/Oxyz.txt", Oxygen_xyz)
        np.savetxt(ifolder + "/Config" + str(iconfig) + "/Hxyz.txt", H_xyz)
        np.savetxt(ifolder + "/Config" + str(iconfig) + "/wxyz.txt", wannier_xyz)
        np.savetxt(ifolder + "/Config" + str(iconfig) + "/box.txt", np.array([boxlength]))
        np.savetxt(ifolder + "/Config" + str(iconfig) + "/fO.txt", fO_total)
        np.savetxt(ifolder + "/Config" + str(iconfig) + "/fH.txt", fH_total)

wxyz_stack = np.stack(wxyz_all, axis=-1).reshape((nwannier,  3, nfolders, nconfigs))
Oxyz_stack = np.stack(Oxyz_all, axis=-1).reshape((noxygen, 3, nfolders, nconfigs))
Hxyz_stack = np.stack(Hxyz_all, axis=-1).reshape((nhydrogen, 3, nfolders, nconfigs))
boxlength_stack = np.stack(boxlength_all).reshape((nfolders, nconfigs))

fO_stack = np.stack(fO_all, axis=-1).reshape((noxygen, 3, nfolders, nconfigs))
fH_stack = np.stack(fH_all, axis=-1).reshape((nhydrogen, 3, nfolders, nconfigs))


np.save("wxyz_allconfigs", wxyz_stack)
np.save("Oxyz_allconfigs", Oxyz_stack)
np.save("Hxyz_allconfigs", Hxyz_stack)
np.save("boxlength_allconfigs", boxlength_stack)
np.save("fO_allconfigs", fO_stack)
np.save("fH_allconfigs", fH_stack)


# assemble the GT features
def assemble_features(feature_name,  id_center):
    if id_center == "O":
        ncenter = noxygen
    else:
        ncenter = natoms - noxygen

    features_all = ()
    features_d_all = ()
    for iconfig in good_configs:
        features = np.loadtxt("D0/Config" + str(iconfig) + "/features_" + feature_name + ".txt", dtype=np.float32)
        features_all += (features, )
        features_d = np.loadtxt("D0/Config" + str(iconfig) + "/features_d" + feature_name + ".txt", dtype=np.float32)
        features_d_all += (features_d.reshape((features.shape[0], ncenter, natoms, 3)), )

    features_all = np.transpose(np.stack(features_all, axis=0), axes=(0, 2, 1))  # now it is (nconfig, ncenter, nfeatures)
    features_d_all = np.transpose(np.stack(features_d_all, axis=0), axes=(0, 2, 3, 4, 1))  # now it is (nconfig, ncenter, natoms, 3, nfeatures)

    np.save("features_" + feature_name, features_all)
    np.save("features_d" + feature_name, features_d_all)

assemble_features("G2OO",  "O")
assemble_features("G2OH",  "O")
assemble_features("G2HO",  "H")
assemble_features("G2HH",  "H")

assemble_features("G4OOO",  "O")
assemble_features("G4OOH",  "O")
assemble_features("G4OHH",  "O")
assemble_features("G4HHO",  "H")
assemble_features("G4HOO",  "H")

def assemble_features_further(feature_name_tuple):  # assemble features that have the same center atom
    features_all = ()
    features_d_all = ()
    for feature_name in feature_name_tuple:
        features = np.load("features_" + feature_name + ".npy")
        features_d = np.load("features_d" + feature_name + ".npy")
        features_all += (features,)
        features_d_all += (features_d,)

    features_all = np.concatenate(features_all, axis=-1)  # stack along the nfeatures axis
    features_d_all = np.transpose(np.concatenate(features_d_all, axis=-1), axes=(0, 2, 3, 1, 4))
    # stack along the nfeatures axis, then make sure the number of center atoms is at the second last axis

    return features_all, features_d_all


xO, xO_d = assemble_features_further(
    ("G2OO", "G2OH", "G4OOO", "G4OOH", "G4OHH"))  # all the O centered features and derivatives
xH, xH_d = assemble_features_further(("G2HH", "G2HO", "G4HHO", "G4HOO"))  # all the H centered features and derivatives


xOO_d = xO_d[:, :noxygen, :, :, :]  # the effect of the move of O atoms on the O features
xHO_d = xO_d[:, noxygen:, :, :, :]  # the effect of the move of H atoms on the O features
xHH_d = xH_d[:, noxygen:, :, :, :]  # the effect of the move of H atoms on the H features
xOH_d = xH_d[:, :noxygen, :, :, :]  # the effect of the move of O atoms on the H features

xO_av = np.mean(xO, axis=(0, 1))
xO_min = np.min(xO, axis=(0, 1))
xO_max = np.max(xO, axis=(0, 1))
np.savetxt("xO_scalefactor.txt", np.stack((xO_av, xO_min, xO_max), axis=-1))

xH_av = np.mean(xH, axis=(0, 1))
xH_min = np.min(xH, axis=(0, 1))
xH_max = np.max(xH, axis=(0, 1))
np.savetxt("xH_scalefactor.txt", np.stack((xH_av, xH_min, xH_max), axis=-1))

xO = (xO - xO_av) / (xO_max - xO_min)
xH = (xH - xH_av) / (xH_max - xH_min)

xOO_d = xOO_d / (xO_max - xO_min)
xHO_d = xHO_d / (xO_max - xO_min)
xHH_d = xHH_d / (xH_max - xH_min)
xOH_d = xOH_d / (xH_max - xH_min)

np.save("xO", xO)
np.save("xH", xH)
np.save("xOO_d", xOO_d)
np.save("xHO_d", xHO_d)
np.save("xHH_d", xHH_d)
np.save("xOH_d", xOH_d)

