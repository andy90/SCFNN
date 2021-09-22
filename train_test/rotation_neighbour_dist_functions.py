# those frequently used functions that is used to get distance, nearest neighbours, and perform rotations
import numpy as np
from parameters import *

from numpy.linalg import norm

def get_rotation(x, y):
    nx = x / norm(x)
    y = y / norm(y)
    new_ax = np.cross(nx, y) / norm(np.cross(nx, y))

    phi = np.arccos(np.dot(nx, y))

    W_flatten = np.array([0, new_ax[2], -new_ax[1], -new_ax[2], 0, new_ax[0], new_ax[1], -new_ax[0], 0])
    W = W_flatten.reshape((3, 3), order="F")

    Ro = np.identity(3) + np.sin(phi) * W + 2 * np.sin(phi / 2) ** 2 * np.dot(W, W)
    return Ro

def get_neighbour_rij(nxyz, wcxyz, boxlength, nneighbour):
    nxyz_expand = np.expand_dims(nxyz, axis=1)  # expand the j indices. Now dim1 is natoms, dim2 is 1, dim3 is xyz
    wcxyz_expand = np.expand_dims(wcxyz, axis=0)  # expand the i indices. Now dim1 is 1, dim2 is nwannier, dim3 is xyz
    vecrij = wcxyz_expand - nxyz_expand - np.round((wcxyz_expand - nxyz_expand) / boxlength) * boxlength  # dim1 is natoms, dim2 is nwanniers, dim3 is xyz
    rij = np.linalg.norm(vecrij, axis=2)

    rij_indmin = np.argsort(rij, axis=1)
    i_indexarray = np.expand_dims(np.arange(rij.shape[0]), axis=1)

    i_indexarray2 = np.expand_dims(np.arange(rij.shape[0]), axis=(1, 2))
    ixyz_indexarray2 = np.expand_dims(np.arange(3), axis=(0, 1))
    rij_indmin2 = np.expand_dims(rij_indmin, axis=2)

    rij_min = rij[i_indexarray, rij_indmin]
    vecrij_min = vecrij[i_indexarray2, rij_indmin2, ixyz_indexarray2]

    return rij_indmin[:, 0:nneighbour], rij_min[:, 0:nneighbour], vecrij_min[:, 0:nneighbour, :]

def get_Oneighbor(Oxyz_all, Hxyz_all, boxlength_all): # get the rOH1 and rOH2 centered around the Oxygen
    Hind_all = ()
    rOH_all = ()
    vecrOH_all = ()
    for ifolder in range(nfolders):
        for iconfig in range(nconfigs):
            data_OH = get_neighbour_rij(Oxyz_all[:, :, ifolder, iconfig], Hxyz_all[:, :, ifolder, iconfig], boxlength_all[ifolder, iconfig], 2)
            Hind_all += (data_OH[0], )
            rOH_all += (data_OH[1], )
            vecrOH_all += (data_OH[2], )

    Hind_stack = np.stack(Hind_all, axis=-1).reshape((noxygen, 2, nfolders, nconfigs))
    rOH_stack = np.stack(rOH_all, axis=-1).reshape((noxygen, 2, nfolders, nconfigs))
    vecrOH_stack = np.stack(vecrOH_all, axis=-1).reshape((noxygen, 2, 3, nfolders, nconfigs))
    return Hind_stack, rOH_stack, vecrOH_stack


def get_Hneighbor(Oxyz_all, Hxyz_all, boxlength_all): # get the rOH1 and rOH2 centered around the Oxygen
    Oind_all = ()
    rHO_all = ()
    vecrHO_all = ()

    Hind_all = ()
    rHH_all = ()
    vecrHH_all = ()
    for ifolder in range(nfolders):
        for iconfig in range(nconfigs):
            data_HO = get_neighbour_rij(Hxyz_all[:, :, ifolder, iconfig], Oxyz_all[:, :, ifolder, iconfig], boxlength_all[ifolder, iconfig], 1)
            Oind_all += (data_HO[0], )
            rHO_all += (data_HO[1], )
            vecrHO_all += (data_HO[2], )

            data_HH = get_neighbour_rij(Hxyz_all[:, :, ifolder, iconfig], Hxyz_all[:, :, ifolder, iconfig],
                                        boxlength_all[ifolder, iconfig], 2)
            Hind_all += (data_HH[0][:, 1],)
            rHH_all += (data_HH[1][:, 1],)
            vecrHH_all += (data_HH[2][:, 1, :],)

    Oind_stack = np.stack(Oind_all, axis=-1).reshape((nhydrogen, 1, nfolders, nconfigs))
    rHO_stack = np.stack(rHO_all, axis=-1).reshape((nhydrogen, 1, nfolders, nconfigs))
    vecrHO_stack = np.stack(vecrHO_all, axis=-1).reshape((nhydrogen, 1, 3, nfolders, nconfigs))

    Hind_stack = np.stack(Hind_all, axis=-1).reshape((nhydrogen, 1, nfolders, nconfigs))
    rHH_stack = np.stack(rHH_all, axis=-1).reshape((nhydrogen, 1, nfolders, nconfigs))
    vecrHH_stack = np.stack(vecrHH_all, axis=-1).reshape((nhydrogen, 1, 3, nfolders, nconfigs))

    OHind_all = np.concatenate((Oind_stack, Hind_stack), axis=1)
    rHHO_all = np.concatenate((rHO_stack, rHH_stack), axis=1)
    vecrHHO_all = np.concatenate((vecrHO_stack, vecrHH_stack), axis=1)
    return OHind_all, rHHO_all, vecrHHO_all



def generate_rotamers(rOH_nn):
    z_axis = np.array([0, 0, 1])
    y_axis = np.array([0, 1, 0])
    rotamers1 = np.zeros((rOH_nn.shape[0], 3, 3, nfolders, nconfigs), dtype=np.float32)
    rotamers2 = np.zeros((rOH_nn.shape[0], 3, 3, nfolders, nconfigs), dtype=np.float32)
    for ifolder in range(nfolders):
        for iconfig in range(nconfigs):
            for io in range(rOH_nn.shape[0]):
                Ro1 = get_rotation(rOH_nn[io, 0, :, ifolder, iconfig], z_axis)
                rH2Op = Ro1 @ rOH_nn[io, 1, :, ifolder, iconfig]
                Ro2 = get_rotation(np.array([rH2Op[0], rH2Op[1], 0]), y_axis)
                rotamers1[io, :, :, ifolder, iconfig] = Ro1
                rotamers2[io, :, :, ifolder, iconfig] = Ro2


    return rotamers1, rotamers2

def rotate(xyz, rotamers1, rotamers2):
    xyz_expand = np.expand_dims(xyz, axis=1)
    rotamers1_expand = np.expand_dims(rotamers1, axis=1)
    rotamers2_expand = np.expand_dims(rotamers2, axis=1)
    xyz_rotated = np.sum(rotamers2_expand * np.expand_dims(np.sum(rotamers1_expand * xyz_expand, axis=3), axis=2), axis=3)

    return xyz_rotated

def shift_rotate(xyz, Originxyz, boxlength_all, rotamers1, rotamers2):
    xyz_expand = np.expand_dims(xyz, axis=0)
    Oxyz_expand = np.expand_dims(Originxyz, axis=1)
    xyz_shift = xyz_expand - Oxyz_expand - np.round((xyz_expand - Oxyz_expand)/boxlength_all)*boxlength_all
    xyz_shift_expand = np.expand_dims(xyz_shift, axis=2)
    rotamers1_expand = np.expand_dims(rotamers1, axis=1)
    rotamers2_expand = np.expand_dims(rotamers2, axis=1)
    xyz_rotated = np.sum(rotamers2_expand * np.expand_dims(np.sum(rotamers1_expand * xyz_shift_expand, axis=3), axis=2), axis=3)

    return xyz_rotated

def backrotate(xyz, rotamers1, rotamers2):  # only rotate force, or a single origin rotated configure
    xyz_expand = np.expand_dims(xyz, axis=2)
    xyz_original = np.sum(rotamers1 * np.expand_dims(np.sum(rotamers2 * xyz_expand, axis=1), axis=2), axis=1)
    return xyz_original

def backrotate_shift(xyz, Originxyz, rotamers1, rotamers2):
    xyz_expand = np.expand_dims(xyz, axis=3)
    rotamers1_expand = np.expand_dims(rotamers1, axis=1)
    rotamers2_expand = np.expand_dims(rotamers2, axis=1)
    Oxyz_expand = np.expand_dims(Originxyz, axis=1)
    xyz_original =  Oxyz_expand + np.sum(rotamers1_expand * np.expand_dims(np.sum(rotamers2_expand * xyz_expand, axis=2), axis=3), axis=2)

    return xyz_original

def map_wxyz(wxyz_all_rotated): # find out the 4 closest neighbours of the wannier center corresponding to each oxygen
    rij = np.linalg.norm(wxyz_all_rotated, axis=2)
    rij_indmin = np.argsort(rij, axis=1)
    i_indexarray = np.expand_dims(np.arange(rij.shape[0]), axis=(1, 2, 3))
    ifolder_indexarray = np.expand_dims(np.arange(nfolders), axis=(0, 1, 3))
    iconfig_indexarray = np.expand_dims(np.arange(nconfigs), axis=(0, 1, 2))

    rij_indmin2 = np.expand_dims(rij_indmin, axis=2)
    i_indexarray2 = np.expand_dims(np.arange(rij.shape[0]), axis=(1, 2, 3, 4))
    ifolder_indexarray2 = np.expand_dims(np.arange(nfolders), axis=(0, 1, 2, 4))
    iconfig_indexarray2 = np.expand_dims(np.arange(nconfigs), axis=(0, 1, 2, 3))
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

