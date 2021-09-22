# produce features
import numpy as np
from numpy import vectorize
from parameters import *

# the cutoff function, it decays to 0 at rc
def fc(rij):
    rc = 12
    if rij <= rc:
        y = np.power(np.tanh(1 - rij / rc), 3)
    else:
        y = 0.0
    return y


vfc = vectorize(fc)  # vectorized fc function


# type 2 symmetry function for all possible nucleus i
def G2(rij, yeta, rs):
    return np.exp(- yeta * np.square(rij - rs)) * vfc(rij)

# type 4 symmetry function for all possible nucleus i
def G4(rij, cosalpha, zeta, yeta, lam):
    y = np.exp(- yeta * rij ** 2) * vfc(rij) * (1 + lam * cosalpha) ** zeta
    y = y * 2 ** (1 - zeta)
    return y

# type 2 symmetry function for all possible nucleus i multiplied by the external field
def G2E(rij, yeta, rs, E):
    return E * np.exp(- yeta * np.square(rij - rs)) * vfc(rij)

# type 4 symmetry function for all possible nucleus i multiplied by the external field
def G4E(rij, cosalpha, zeta, yeta, lam, E):
    y = E * np.exp(- yeta * rij ** 2) * vfc(rij) * (1 + lam * cosalpha) ** zeta
    y = y * 2 ** (1 - zeta)
    return y

# type 2 symmetry function for all possible nucleus i multiplied by the external field and the displacement to i
def G2P(rij, yeta, rs, E, vecrij):
    return vecrij * E * np.exp(- yeta * np.square(rij - rs)) * vfc(rij)

def features_wannier_GT(vecrij_OO, vecrij_OH):
    rij_OO = np.linalg.norm(vecrij_OO, axis=2)
    rij_OH = np.linalg.norm(vecrij_OH, axis=2)

    ijones_diag = np.zeros((rij_OO.shape[0], rij_OO.shape[1], 1), dtype=np.float32)
    np.fill_diagonal(ijones_diag[:, :, 0], 100)
    rij_OO += ijones_diag

    cosalpha_OO_xax = vecrij_OO[:, :, 0, :] / rij_OO
    cosalpha_OO_yax = vecrij_OO[:, :, 1, :] / rij_OO
    cosalpha_OO_zax = vecrij_OO[:, :, 2, :] / rij_OO

    cosalpha_OH_xax = vecrij_OH[:, :, 0, :] / rij_OH
    cosalpha_OH_yax = vecrij_OH[:, :, 1, :] / rij_OH
    cosalpha_OH_zax = vecrij_OH[:, :, 2, :] / rij_OH

    parameters2_OO = np.loadtxt("wG2_parameters_OO.txt")
    parameters2_OH = np.loadtxt("wG2_parameters_OH.txt")

    features_G2OO = np.zeros((parameters2_OO.shape[0], noxygen, nconfigs_test), dtype=np.float32)
    features_G2OH = np.zeros((parameters2_OH.shape[0], noxygen, nconfigs_test), dtype=np.float32)

    parameters4_OO = np.loadtxt("wG4_parameters_OO.txt")
    parameters4_OH = np.loadtxt("wG4_parameters_OH.txt")

    features_G4OO_zax = np.zeros((parameters4_OO.shape[0], noxygen, nconfigs_test), dtype=np.float32)
    features_G4OH_zax = np.zeros((parameters4_OH.shape[0], noxygen, nconfigs_test), dtype=np.float32)

    features_G4OO_xax = np.zeros((parameters4_OO.shape[0], noxygen, nconfigs_test), dtype=np.float32)
    features_G4OH_xax = np.zeros((parameters4_OH.shape[0], noxygen, nconfigs_test), dtype=np.float32)

    features_G4OO_yax = np.zeros((parameters4_OO.shape[0], noxygen, nconfigs_test), dtype=np.float32)
    features_G4OH_yax = np.zeros((parameters4_OH.shape[0], noxygen, nconfigs_test), dtype=np.float32)

    for ip in range(parameters2_OO.shape[0]):
        rs = parameters2_OO[ip, 0]
        yeta = parameters2_OO[ip, 1]

        G2_ip = G2(rij_OO, yeta, rs)

        # get the final G20O
        G2_ip_sum = np.sum(G2_ip, axis=1)

        features_G2OO[ip, :, :] = G2_ip_sum[:, :]

    for ip in range(parameters2_OH.shape[0]):
        rs = parameters2_OH[ip, 0]
        yeta = parameters2_OH[ip, 1]

        G2_ip = G2(rij_OH, yeta, rs)

        # get the final G20H
        G2_ip_sum = np.sum(G2_ip, axis=1)

        features_G2OH[ip, :, :] = G2_ip_sum[:, :]

    for ip in range(parameters4_OO.shape[0]):
        rs = parameters4_OO[ip, 0]
        yeta = parameters4_OO[ip, 1]
        lam = parameters4_OO[ip, 2]
        zeta = parameters4_OO[ip, 3]

        G4_ip = G4(rij_OO, cosalpha_OO_zax, zeta, yeta, lam)

        G4_ip_sum = np.sum(G4_ip, axis=1)

        features_G4OO_zax[ip, :, :] = G4_ip_sum[:, :]

    for ip in range(parameters4_OO.shape[0]):
        rs = parameters4_OO[ip, 0]
        yeta = parameters4_OO[ip, 1]
        lam = parameters4_OO[ip, 2]
        zeta = parameters4_OO[ip, 3]

        G4_ip = G4(rij_OO, cosalpha_OO_xax, zeta, yeta, lam)

        G4_ip_sum = np.sum(G4_ip, axis=1)

        features_G4OO_xax[ip, :, :] = G4_ip_sum[:, :]

    for ip in range(parameters4_OO.shape[0]):
        rs = parameters4_OO[ip, 0]
        yeta = parameters4_OO[ip, 1]
        lam = parameters4_OO[ip, 2]
        zeta = parameters4_OO[ip, 3]

        G4_ip = G4(rij_OO, cosalpha_OO_yax, zeta, yeta, lam)

        G4_ip_sum = np.sum(G4_ip, axis=1)

        features_G4OO_yax[ip, :, :] = G4_ip_sum[:, :]

    for ip in range(parameters4_OH.shape[0]):
        rs = parameters4_OH[ip, 0]
        yeta = parameters4_OH[ip, 1]
        lam = parameters4_OH[ip, 2]
        zeta = parameters4_OH[ip, 3]

        G4_ip = G4(rij_OH, cosalpha_OH_zax, zeta, yeta, lam)

        G4_ip_sum = np.sum(G4_ip, axis=1)

        features_G4OH_zax[ip, :, :] = G4_ip_sum[:, :]

    for ip in range(parameters4_OH.shape[0]):
        rs = parameters4_OH[ip, 0]
        yeta = parameters4_OH[ip, 1]
        lam = parameters4_OH[ip, 2]
        zeta = parameters4_OH[ip, 3]

        G4_ip = G4(rij_OH, cosalpha_OH_xax, zeta, yeta, lam)

        G4_ip_sum = np.sum(G4_ip, axis=1)

        features_G4OH_xax[ip, :, :] = G4_ip_sum[:, :]

    for ip in range(parameters4_OH.shape[0]):
        rs = parameters4_OH[ip, 0]
        yeta = parameters4_OH[ip, 1]
        lam = parameters4_OH[ip, 2]
        zeta = parameters4_OH[ip, 3]

        G4_ip = G4(rij_OH, cosalpha_OH_yax, zeta, yeta, lam)

        G4_ip_sum = np.sum(G4_ip, axis=1)

        features_G4OH_yax[ip, :, :] = G4_ip_sum[:, :]

    features_total = np.concatenate((features_G2OO, features_G2OH,  features_G4OO_xax, features_G4OO_yax, features_G4OO_zax, features_G4OH_xax, features_G4OH_yax, features_G4OH_zax,), axis=0)

    features_av = np.expand_dims(np.mean(features_total, axis=(1,2)), axis=(1,2))
    features_min = np.expand_dims(np.min(features_total, axis=(1,2)), axis=(1,2))
    features_max = np.expand_dims(np.max(features_total, axis=(1,2)), axis=(1,2))

    features_scaled = (features_total - features_av) / (features_max - features_min)

    scalefactor = np.stack((np.mean(features_total, axis=(1,2)), np.min(features_total, axis=(1,2)), np.max(features_total, axis=(1,2))), axis=-1)
    return features_scaled


def features_wannier_peturb(vecrij_OO, vecrij_OH, EO, EH):
    rij_OO = np.linalg.norm(vecrij_OO, axis=2)
    rij_OH = np.linalg.norm(vecrij_OH, axis=2)

    ijones_diag = np.zeros((rij_OO.shape[0], rij_OO.shape[1], 1, 1), dtype=np.float32)
    np.fill_diagonal(ijones_diag[:, :, 0, 0], 100)
    rij_OO += ijones_diag

    # cosalpha_OO_xax = vecrij_OO[:, :, 0, :, :] / rij_OO
    # cosalpha_OO_yax = vecrij_OO[:, :, 1, :, :] / rij_OO
    # cosalpha_OO_zax = vecrij_OO[:, :, 2, :, :] / rij_OO

    # cosalpha_OH_xax = vecrij_OH[:, :, 0, :, :] / rij_OH
    # cosalpha_OH_yax = vecrij_OH[:, :, 1, :, :] / rij_OH
    # cosalpha_OH_zax = vecrij_OH[:, :, 2, :, :] / rij_OH

    rij_OO_e = np.expand_dims(rij_OO, axis=2)
    rij_OH_e = np.expand_dims(rij_OH, axis=2)

    # cosalpha_OO_xax_e = np.expand_dims(cosalpha_OO_xax, axis=2)
    # cosalpha_OO_yax_e = np.expand_dims(cosalpha_OO_yax, axis=2)
    # cosalpha_OO_zax_e = np.expand_dims(cosalpha_OO_zax, axis=2)

    # cosalpha_OH_xax_e = np.expand_dims(cosalpha_OH_xax, axis=2)
    # cosalpha_OH_yax_e = np.expand_dims(cosalpha_OH_yax, axis=2)
    # cosalpha_OH_zax_e = np.expand_dims(cosalpha_OH_zax, axis=2)

    parameters2_OO = np.loadtxt("dwG2_parameters_OO.txt")
    parameters2_OH = np.loadtxt("dwG2_parameters_OH.txt")

    parameters4_OO = np.loadtxt("dwG4_parameters_OO.txt")
    parameters4_OH = np.loadtxt("dwG4_parameters_OH.txt")

    # features_G2OO = np.zeros((parameters2_OO.shape[0], noxygen, 1, nfolders, nconfigs_test), dtype=np.float32)
    # features_G2OH = np.zeros((parameters2_OH.shape[0], noxygen, 1, nfolders, nconfigs_test), dtype=np.float32)

    # features_G4OO_zax = np.zeros((parameters4_OO.shape[0], noxygen, 1, nfolders, nconfigs_test), dtype=np.float32)
    # features_G4OH_zax = np.zeros((parameters4_OH.shape[0], noxygen, 1, nfolders, nconfigs_test), dtype=np.float32)

    # features_G4OO_xax = np.zeros((parameters4_OO.shape[0], noxygen, 1, nfolders, nconfigs_test), dtype=np.float32)
    # features_G4OH_xax = np.zeros((parameters4_OH.shape[0], noxygen, 1, nfolders, nconfigs_test), dtype=np.float32)

    # features_G4OO_yax = np.zeros((parameters4_OO.shape[0], noxygen, 1, nfolders, nconfigs_test), dtype=np.float32)
    # features_G4OH_yax = np.zeros((parameters4_OH.shape[0], noxygen, 1, nfolders, nconfigs_test), dtype=np.float32)

    # features for those fields, we have additional dimension (the third one) which records the xyz of the E field
    features_EG2OO = np.zeros((parameters2_OO.shape[0], noxygen, 3, nfolders, nconfigs_test), dtype=np.float32)
    features_EG2OH = np.zeros((parameters2_OH.shape[0], noxygen, 3, nfolders, nconfigs_test), dtype=np.float32)

    # features_EG4OO_zax = np.zeros((parameters4_OO.shape[0], noxygen, 3, nfolders, nconfigs_test), dtype=np.float32)
    # features_EG4OH_zax = np.zeros((parameters4_OH.shape[0], noxygen, 3, nfolders, nconfigs_test), dtype=np.float32)

    # features_EG4OO_xax = np.zeros((parameters4_OO.shape[0], noxygen, 3, nfolders, nconfigs_test), dtype=np.float32)
    # features_EG4OH_xax = np.zeros((parameters4_OH.shape[0], noxygen, 3, nfolders, nconfigs_test), dtype=np.float32)

    # features_EG4OO_yax = np.zeros((parameters4_OO.shape[0], noxygen, 3, nfolders, nconfigs_test), dtype=np.float32)
    # features_EG4OH_yax = np.zeros((parameters4_OH.shape[0], noxygen, 3, nfolders, nconfigs_test), dtype=np.float32)

    # for ip in range(parameters2_OO.shape[0]):
    #     rs = parameters2_OO[ip, 0]
    #     yeta = parameters2_OO[ip, 1]

    #     G2_ip = G2(rij_OO_e, yeta, rs)

    #     # get the final G20O
    #     G2_ip_sum = np.sum(G2_ip, axis=1)

    #     features_G2OO[ip, :, :, :, :] = G2_ip_sum[:, :, :, :]

    # for ip in range(parameters2_OH.shape[0]):
    #     rs = parameters2_OH[ip, 0]
    #     yeta = parameters2_OH[ip, 1]

    #     G2_ip = G2(rij_OH_e, yeta, rs)

    #     # get the final G20H
    #     G2_ip_sum = np.sum(G2_ip, axis=1)

    #     features_G2OH[ip, :, :, :, :] = G2_ip_sum[:, :, :, :]

    # for ip in range(parameters4_OO.shape[0]):
    #     rs = parameters4_OO[ip, 0]
    #     yeta = parameters4_OO[ip, 1]
    #     lam = parameters4_OO[ip, 2]
    #     zeta = parameters4_OO[ip, 3]

    #     G4_ip = G4(rij_OO_e, cosalpha_OO_zax_e, zeta, yeta, lam)

    #     G4_ip_sum = np.sum(G4_ip, axis=1)

    #     features_G4OO_zax[ip, :, :, :, :] = G4_ip_sum[:, :, :, :]

    # for ip in range(parameters4_OO.shape[0]):
    #     rs = parameters4_OO[ip, 0]
    #     yeta = parameters4_OO[ip, 1]
    #     lam = parameters4_OO[ip, 2]
    #     zeta = parameters4_OO[ip, 3]

    #     G4_ip = G4(rij_OO_e, cosalpha_OO_xax_e, zeta, yeta, lam)

    #     G4_ip_sum = np.sum(G4_ip, axis=1)

    #     features_G4OO_xax[ip, :, :, :, :] = G4_ip_sum[:, :, :, :]

    # for ip in range(parameters4_OO.shape[0]):
    #     rs = parameters4_OO[ip, 0]
    #     yeta = parameters4_OO[ip, 1]
    #     lam = parameters4_OO[ip, 2]
    #     zeta = parameters4_OO[ip, 3]

    #     G4_ip = G4(rij_OO_e, cosalpha_OO_yax_e, zeta, yeta, lam)

    #     G4_ip_sum = np.sum(G4_ip, axis=1)

    #     features_G4OO_yax[ip, :, :, :, :] = G4_ip_sum[:, :, :, :]

    # for ip in range(parameters4_OH.shape[0]):
    #     rs = parameters4_OH[ip, 0]
    #     yeta = parameters4_OH[ip, 1]
    #     lam = parameters4_OH[ip, 2]
    #     zeta = parameters4_OH[ip, 3]

    #     G4_ip = G4(rij_OH_e, cosalpha_OH_zax_e, zeta, yeta, lam)

    #     G4_ip_sum = np.sum(G4_ip, axis=1)

    #     features_G4OH_zax[ip, :, :, :, :] = G4_ip_sum[:, :, :, :]

    # for ip in range(parameters4_OH.shape[0]):
    #     rs = parameters4_OH[ip, 0]
    #     yeta = parameters4_OH[ip, 1]
    #     lam = parameters4_OH[ip, 2]
    #     zeta = parameters4_OH[ip, 3]

    #     G4_ip = G4(rij_OH_e, cosalpha_OH_xax_e, zeta, yeta, lam)

    #     G4_ip_sum = np.sum(G4_ip, axis=1)

    #     features_G4OH_xax[ip, :, :, :, :] = G4_ip_sum[:, :, :, :]

    # for ip in range(parameters4_OH.shape[0]):
    #     rs = parameters4_OH[ip, 0]
    #     yeta = parameters4_OH[ip, 1]
    #     lam = parameters4_OH[ip, 2]
    #     zeta = parameters4_OH[ip, 3]

    #     G4_ip = G4(rij_OH_e, cosalpha_OH_yax_e, zeta, yeta, lam)

    #     G4_ip_sum = np.sum(G4_ip, axis=1)

    #     features_G4OH_yax[ip, :, :, :, :] = G4_ip_sum[:, :, :, :]

    # features for those fields
    for ip in range(parameters2_OO.shape[0]):
        rs = parameters2_OO[ip, 0]
        yeta = parameters2_OO[ip, 1]

        G2_ip = G2E(rij_OO_e, yeta, rs, EO)

        # get the final G20O
        G2_ip_sum = np.sum(G2_ip, axis=1)

        features_EG2OO[ip, :, :, :, :] = G2_ip_sum[:, :, :, :]

    for ip in range(parameters2_OH.shape[0]):
        rs = parameters2_OH[ip, 0]
        yeta = parameters2_OH[ip, 1]

        G2_ip = G2E(rij_OH_e, yeta, rs, EH)

        # get the final G20H
        G2_ip_sum = np.sum(G2_ip, axis=1)

        features_EG2OH[ip, :, :, :, :] = G2_ip_sum[:, :, :, :]

    # for ip in range(parameters4_OO.shape[0]):
    #     rs = parameters4_OO[ip, 0]
    #     yeta = parameters4_OO[ip, 1]
    #     lam = parameters4_OO[ip, 2]
    #     zeta = parameters4_OO[ip, 3]

    #     G4_ip = G4E(rij_OO_e, cosalpha_OO_zax_e, zeta, yeta, lam, EO)

    #     G4_ip_sum = np.sum(G4_ip, axis=1)

    #     features_EG4OO_zax[ip, :, :, :, :] = G4_ip_sum[:, :, :, :]

    # for ip in range(parameters4_OO.shape[0]):
    #     rs = parameters4_OO[ip, 0]
    #     yeta = parameters4_OO[ip, 1]
    #     lam = parameters4_OO[ip, 2]
    #     zeta = parameters4_OO[ip, 3]

    #     G4_ip = G4E(rij_OO_e, cosalpha_OO_xax_e, zeta, yeta, lam, EO)

    #     G4_ip_sum = np.sum(G4_ip, axis=1)

    #     features_EG4OO_xax[ip, :, :, :, :] = G4_ip_sum[:, :, :, :]

    # for ip in range(parameters4_OO.shape[0]):
    #     rs = parameters4_OO[ip, 0]
    #     yeta = parameters4_OO[ip, 1]
    #     lam = parameters4_OO[ip, 2]
    #     zeta = parameters4_OO[ip, 3]

    #     G4_ip = G4E(rij_OO_e, cosalpha_OO_yax_e, zeta, yeta, lam, EO)

    #     G4_ip_sum = np.sum(G4_ip, axis=1)

    #     features_EG4OO_yax[ip, :, :, :, :] = G4_ip_sum[:, :, :, :]

    # for ip in range(parameters4_OH.shape[0]):
    #     rs = parameters4_OH[ip, 0]
    #     yeta = parameters4_OH[ip, 1]
    #     lam = parameters4_OH[ip, 2]
    #     zeta = parameters4_OH[ip, 3]

    #     G4_ip = G4E(rij_OH_e, cosalpha_OH_zax_e, zeta, yeta, lam, EH)

    #     G4_ip_sum = np.sum(G4_ip, axis=1)

    #     features_EG4OH_zax[ip, :, :, :, :] = G4_ip_sum[:, :, :, :]

    # for ip in range(parameters4_OH.shape[0]):
    #     rs = parameters4_OH[ip, 0]
    #     yeta = parameters4_OH[ip, 1]
    #     lam = parameters4_OH[ip, 2]
    #     zeta = parameters4_OH[ip, 3]

    #     G4_ip = G4E(rij_OH_e, cosalpha_OH_xax_e, zeta, yeta, lam, EH)

    #     G4_ip_sum = np.sum(G4_ip, axis=1)

    #     features_EG4OH_xax[ip, :, :, :, :] = G4_ip_sum[:, :, :, :]

    # for ip in range(parameters4_OH.shape[0]):
    #     rs = parameters4_OH[ip, 0]
    #     yeta = parameters4_OH[ip, 1]
    #     lam = parameters4_OH[ip, 2]
    #     zeta = parameters4_OH[ip, 3]

    #     G4_ip = G4E(rij_OH_e, cosalpha_OH_yax_e, zeta, yeta, lam, EH)

    #     G4_ip_sum = np.sum(G4_ip, axis=1)

    #     features_EG4OH_yax[ip, :, :, :, :] = G4_ip_sum[:, :, :, :]

    features_EG2OO_new = np.transpose(features_EG2OO , axes=(0, 2, 1, 3, 4)).reshape(
        (parameters2_OO.shape[0] * 3, noxygen, 3, nconfigs_test))
    features_EG2OH_new = np.transpose(features_EG2OH , axes=(0, 2, 1, 3, 4)).reshape(
        (parameters2_OH.shape[0] * 3, noxygen, 3, nconfigs_test))
    # features_EG4OO_H1O_new = np.transpose(features_EG4OO_zax , axes=(0, 2, 1, 3, 4)).reshape(
    #     (parameters4_OO.shape[0] * 3, noxygen, 3, nconfigs_test))
    # features_EG4OO_xax_new = np.transpose(features_EG4OO_xax , axes=(0, 2, 1, 3, 4)).reshape(
    #     (parameters4_OO.shape[0] * 3, noxygen, 3, nconfigs_test))
    # features_EG4OO_yax_new = np.transpose(features_EG4OO_yax , axes=(0, 2, 1, 3, 4)).reshape(
    #     (parameters4_OO.shape[0] * 3, noxygen, 3, nconfigs_test))
    # features_EG4OH_H1O_new = np.transpose(features_EG4OH_zax , axes=(0, 2, 1, 3, 4)).reshape(
    #     (parameters4_OH.shape[0] * 3, noxygen, 3, nconfigs_test))
    # features_EG4OH_xax_new = np.transpose(features_EG4OH_xax , axes=(0, 2, 1, 3, 4)).reshape(
    #     (parameters4_OH.shape[0] * 3, noxygen, 3, nconfigs_test))
    # features_EG4OH_yax_new = np.transpose(features_EG4OH_yax , axes=(0, 2, 1, 3, 4)).reshape(
    #     (parameters4_OH.shape[0] * 3, noxygen, 3, nconfigs_test))

    features_total = np.concatenate((features_EG2OO_new, features_EG2OH_new), axis=0)

    #features_av = np.expand_dims(np.mean(features_total, axis=(1, 2, 3)), axis=(1, 2, 3))
    #features_std = np.expand_dims(np.std(features_total, axis=(1, 2, 3)), axis=(1, 2, 3))

    return features_total

# this is a general function that produces peturbation features for the rotated force
# it could produce features for both fO and fH
# if produce features for fO, then the name of the inputs follows the fashion in the definition of the function
# if produce features for fH, then switch O with H.
def features_force_peturb(vecrij_OO, vecrij_OH, EO, EH, parameters2_OO, parameters2_OH, parameters4_OO, parameters4_OH):
    rij_OO = np.linalg.norm(vecrij_OO, axis=2)
    rij_OH = np.linalg.norm(vecrij_OH, axis=2)

    ind_zero = rij_OO == 0
    rij_OO[ind_zero] = rij_OO[ind_zero] + 100

    # cosalpha_OO_xax = vecrij_OO[:, :, 0, :, :] / rij_OO
    # cosalpha_OO_yax = vecrij_OO[:, :, 1, :, :] / rij_OO
    # cosalpha_OO_zax = vecrij_OO[:, :, 2, :, :] / rij_OO

    # cosalpha_OH_xax = vecrij_OH[:, :, 0, :, :] / rij_OH
    # cosalpha_OH_yax = vecrij_OH[:, :, 1, :, :] / rij_OH
    # cosalpha_OH_zax = vecrij_OH[:, :, 2, :, :] / rij_OH

    rij_OO_e = np.expand_dims(rij_OO, axis=2)
    rij_OH_e = np.expand_dims(rij_OH, axis=2)

    # cosalpha_OO_xax_e = np.expand_dims(cosalpha_OO_xax, axis=2)
    # cosalpha_OO_yax_e = np.expand_dims(cosalpha_OO_yax, axis=2)
    # cosalpha_OO_zax_e = np.expand_dims(cosalpha_OO_zax, axis=2)

    # cosalpha_OH_xax_e = np.expand_dims(cosalpha_OH_xax, axis=2)
    # cosalpha_OH_yax_e = np.expand_dims(cosalpha_OH_yax, axis=2)
    # cosalpha_OH_zax_e = np.expand_dims(cosalpha_OH_zax, axis=2)

    # features_G2OO = np.zeros((parameters2_OO.shape[0], rij_OO.shape[0], 1, nfolders, nconfigs_test), dtype=np.float32)
    # features_G2OH = np.zeros((parameters2_OH.shape[0], rij_OO.shape[0], 1, nfolders, nconfigs_test), dtype=np.float32)

    # features_G4OO_zax = np.zeros((parameters4_OO.shape[0], rij_OO.shape[0], 1, nfolders, nconfigs_test), dtype=np.float32)
    # features_G4OH_zax = np.zeros((parameters4_OH.shape[0], rij_OO.shape[0], 1, nfolders, nconfigs_test), dtype=np.float32)

    # features_G4OO_xax = np.zeros((parameters4_OO.shape[0], rij_OO.shape[0], 1, nfolders, nconfigs_test), dtype=np.float32)
    # features_G4OH_xax = np.zeros((parameters4_OH.shape[0], rij_OO.shape[0], 1, nfolders, nconfigs_test), dtype=np.float32)

    # features_G4OO_yax = np.zeros((parameters4_OO.shape[0], rij_OO.shape[0], 1, nfolders, nconfigs_test), dtype=np.float32)
    # features_G4OH_yax = np.zeros((parameters4_OH.shape[0], rij_OO.shape[0], 1, nfolders, nconfigs_test), dtype=np.float32)

    # features for those fields, we have additional dimension (the third one) which records the xyz of the E field
    features_EG2OO = np.zeros((parameters2_OO.shape[0], rij_OO.shape[0], 3, nfolders, nconfigs_test), dtype=np.float32)
    features_EG2OH = np.zeros((parameters2_OH.shape[0], rij_OO.shape[0], 3, nfolders, nconfigs_test), dtype=np.float32)

    # features_EG4OO_zax = np.zeros((parameters4_OO.shape[0], rij_OO.shape[0], 3, nfolders, nconfigs_test), dtype=np.float32)
    # features_EG4OH_zax = np.zeros((parameters4_OH.shape[0], rij_OO.shape[0], 3, nfolders, nconfigs_test), dtype=np.float32)

    # features_EG4OO_xax = np.zeros((parameters4_OO.shape[0], rij_OO.shape[0], 3, nfolders, nconfigs_test), dtype=np.float32)
    # features_EG4OH_xax = np.zeros((parameters4_OH.shape[0], rij_OO.shape[0], 3, nfolders, nconfigs_test), dtype=np.float32)

    # features_EG4OO_yax = np.zeros((parameters4_OO.shape[0], rij_OO.shape[0], 3, nfolders, nconfigs_test), dtype=np.float32)
    # features_EG4OH_yax = np.zeros((parameters4_OH.shape[0], rij_OO.shape[0], 3, nfolders, nconfigs_test), dtype=np.float32)


    # for ip in range(parameters2_OO.shape[0]):
    #     rs = parameters2_OO[ip, 0]
    #     yeta = parameters2_OO[ip, 1]

    #     G2_ip = G2(rij_OO_e, yeta, rs)

    #     # get the final G20O
    #     G2_ip_sum = np.sum(G2_ip, axis=1)

    #     features_G2OO[ip, :, :, :, :] = G2_ip_sum[:, :, :, :]

    # for ip in range(parameters2_OH.shape[0]):
    #     rs = parameters2_OH[ip, 0]
    #     yeta = parameters2_OH[ip, 1]

    #     G2_ip = G2(rij_OH_e, yeta, rs)

    #     # get the final G20H
    #     G2_ip_sum = np.sum(G2_ip, axis=1)

    #     features_G2OH[ip, :, :, :, :] = G2_ip_sum[:, :, :, :]

    # for ip in range(parameters4_OO.shape[0]):
    #     rs = parameters4_OO[ip, 0]
    #     yeta = parameters4_OO[ip, 1]
    #     lam = parameters4_OO[ip, 2]
    #     zeta = parameters4_OO[ip, 3]

    #     G4_ip = G4(rij_OO_e, cosalpha_OO_zax_e, zeta, yeta, lam)

    #     G4_ip_sum = np.sum(G4_ip, axis=1)

    #     features_G4OO_zax[ip, :, :, :, :] = G4_ip_sum[:, :, :, :]

    # for ip in range(parameters4_OO.shape[0]):
    #     rs = parameters4_OO[ip, 0]
    #     yeta = parameters4_OO[ip, 1]
    #     lam = parameters4_OO[ip, 2]
    #     zeta = parameters4_OO[ip, 3]

    #     G4_ip = G4(rij_OO_e, cosalpha_OO_xax_e, zeta, yeta, lam)

    #     G4_ip_sum = np.sum(G4_ip, axis=1)

    #     features_G4OO_xax[ip, :, :, :, :] = G4_ip_sum[:, :, :, :]

    # for ip in range(parameters4_OO.shape[0]):
    #     rs = parameters4_OO[ip, 0]
    #     yeta = parameters4_OO[ip, 1]
    #     lam = parameters4_OO[ip, 2]
    #     zeta = parameters4_OO[ip, 3]

    #     G4_ip = G4(rij_OO_e, cosalpha_OO_yax_e, zeta, yeta, lam)

    #     G4_ip_sum = np.sum(G4_ip, axis=1)

    #     features_G4OO_yax[ip, :, :, :, :] = G4_ip_sum[:, :, :, :]

    # for ip in range(parameters4_OH.shape[0]):
    #     rs = parameters4_OH[ip, 0]
    #     yeta = parameters4_OH[ip, 1]
    #     lam = parameters4_OH[ip, 2]
    #     zeta = parameters4_OH[ip, 3]

    #     G4_ip = G4(rij_OH_e, cosalpha_OH_zax_e, zeta, yeta, lam)

    #     G4_ip_sum = np.sum(G4_ip, axis=1)

    #     features_G4OH_zax[ip, :, :, :, :] = G4_ip_sum[:, :, :, :]

    # for ip in range(parameters4_OH.shape[0]):
    #     rs = parameters4_OH[ip, 0]
    #     yeta = parameters4_OH[ip, 1]
    #     lam = parameters4_OH[ip, 2]
    #     zeta = parameters4_OH[ip, 3]

    #     G4_ip = G4(rij_OH_e, cosalpha_OH_xax_e, zeta, yeta, lam)

    #     G4_ip_sum = np.sum(G4_ip, axis=1)

    #     features_G4OH_xax[ip, :, :, :, :] = G4_ip_sum[:, :, :, :]

    # for ip in range(parameters4_OH.shape[0]):
    #     rs = parameters4_OH[ip, 0]
    #     yeta = parameters4_OH[ip, 1]
    #     lam = parameters4_OH[ip, 2]
    #     zeta = parameters4_OH[ip, 3]

    #     G4_ip = G4(rij_OH_e, cosalpha_OH_yax_e, zeta, yeta, lam)

    #     G4_ip_sum = np.sum(G4_ip, axis=1)

    #     features_G4OH_yax[ip, :, :, :, :] = G4_ip_sum[:, :, :, :]

    # features for those fields
    for ip in range(parameters2_OO.shape[0]):
        rs = parameters2_OO[ip, 0]
        yeta = parameters2_OO[ip, 1]

        G2_ip = G2E(rij_OO_e, yeta, rs, EO)

        # get the final G20O
        G2_ip_sum = np.sum(G2_ip, axis=1)

        features_EG2OO[ip, :, :, :, :] = G2_ip_sum[:, :, :, :]

    for ip in range(parameters2_OH.shape[0]):
        rs = parameters2_OH[ip, 0]
        yeta = parameters2_OH[ip, 1]

        G2_ip = G2E(rij_OH_e, yeta, rs, EH)

        # get the final G20H
        G2_ip_sum = np.sum(G2_ip, axis=1)

        features_EG2OH[ip, :, :, :, :] = G2_ip_sum[:, :, :, :]

    # for ip in range(parameters4_OO.shape[0]):
    #     rs = parameters4_OO[ip, 0]
    #     yeta = parameters4_OO[ip, 1]
    #     lam = parameters4_OO[ip, 2]
    #     zeta = parameters4_OO[ip, 3]

    #     G4_ip = G4E(rij_OO_e, cosalpha_OO_zax_e, zeta, yeta, lam, EO)

    #     G4_ip_sum = np.sum(G4_ip, axis=1)

    #     features_EG4OO_zax[ip, :, :, :, :] = G4_ip_sum[:, :, :, :]

    # for ip in range(parameters4_OO.shape[0]):
    #     rs = parameters4_OO[ip, 0]
    #     yeta = parameters4_OO[ip, 1]
    #     lam = parameters4_OO[ip, 2]
    #     zeta = parameters4_OO[ip, 3]

    #     G4_ip = G4E(rij_OO_e, cosalpha_OO_xax_e, zeta, yeta, lam, EO)

    #     G4_ip_sum = np.sum(G4_ip, axis=1)

    #     features_EG4OO_xax[ip, :, :, :, :] = G4_ip_sum[:, :, :, :]

    # for ip in range(parameters4_OO.shape[0]):
    #     rs = parameters4_OO[ip, 0]
    #     yeta = parameters4_OO[ip, 1]
    #     lam = parameters4_OO[ip, 2]
    #     zeta = parameters4_OO[ip, 3]

    #     G4_ip = G4E(rij_OO_e, cosalpha_OO_yax_e, zeta, yeta, lam, EO)

    #     G4_ip_sum = np.sum(G4_ip, axis=1)

    #     features_EG4OO_yax[ip, :, :, :, :] = G4_ip_sum[:, :, :, :]

    # for ip in range(parameters4_OH.shape[0]):
    #     rs = parameters4_OH[ip, 0]
    #     yeta = parameters4_OH[ip, 1]
    #     lam = parameters4_OH[ip, 2]
    #     zeta = parameters4_OH[ip, 3]

    #     G4_ip = G4E(rij_OH_e, cosalpha_OH_zax_e, zeta, yeta, lam, EH)

    #     G4_ip_sum = np.sum(G4_ip, axis=1)

    #     features_EG4OH_zax[ip, :, :, :, :] = G4_ip_sum[:, :, :, :]

    # for ip in range(parameters4_OH.shape[0]):
    #     rs = parameters4_OH[ip, 0]
    #     yeta = parameters4_OH[ip, 1]
    #     lam = parameters4_OH[ip, 2]
    #     zeta = parameters4_OH[ip, 3]

    #     G4_ip = G4E(rij_OH_e, cosalpha_OH_xax_e, zeta, yeta, lam, EH)

    #     G4_ip_sum = np.sum(G4_ip, axis=1)

    #     features_EG4OH_xax[ip, :, :, :, :] = G4_ip_sum[:, :, :, :]

    # for ip in range(parameters4_OH.shape[0]):
    #     rs = parameters4_OH[ip, 0]
    #     yeta = parameters4_OH[ip, 1]
    #     lam = parameters4_OH[ip, 2]
    #     zeta = parameters4_OH[ip, 3]

    #     G4_ip = G4E(rij_OH_e, cosalpha_OH_yax_e, zeta, yeta, lam, EH)

    #     G4_ip_sum = np.sum(G4_ip, axis=1)

    #     features_EG4OH_yax[ip, :, :, :, :] = G4_ip_sum[:, :, :, :]

    features_EG2OO_new = np.transpose(features_EG2OO, axes=(0, 2, 1, 3, 4)).reshape(
        (parameters2_OO.shape[0] * 3, rij_OO.shape[0], 3, nconfigs_test))
    features_EG2OH_new = np.transpose(features_EG2OH, axes=(0, 2, 1, 3, 4)).reshape(
        (parameters2_OH.shape[0] * 3, rij_OO.shape[0], 3, nconfigs_test))
    # features_EG4OO_H1O_new = np.transpose(features_EG4OO_zax, axes=(0, 2, 1, 3, 4)).reshape(
    #     (parameters4_OO.shape[0] * 3, rij_OO.shape[0], 3, nconfigs_test))
    # features_EG4OO_xax_new = np.transpose(features_EG4OO_xax, axes=(0, 2, 1, 3, 4)).reshape(
    #     (parameters4_OO.shape[0] * 3, rij_OO.shape[0], 3, nconfigs_test))
    # features_EG4OO_yax_new = np.transpose(features_EG4OO_yax, axes=(0, 2, 1, 3, 4)).reshape(
    #     (parameters4_OO.shape[0] * 3, rij_OO.shape[0], 3, nconfigs_test))
    # features_EG4OH_H1O_new = np.transpose(features_EG4OH_zax, axes=(0, 2, 1, 3, 4)).reshape(
    #     (parameters4_OH.shape[0] * 3, rij_OO.shape[0], 3, nconfigs_test))
    # features_EG4OH_xax_new = np.transpose(features_EG4OH_xax, axes=(0, 2, 1, 3, 4)).reshape(
    #     (parameters4_OH.shape[0] * 3, rij_OO.shape[0], 3, nconfigs_test))
    # features_EG4OH_yax_new = np.transpose(features_EG4OH_yax, axes=(0, 2, 1, 3, 4)).reshape(
    #     (parameters4_OH.shape[0] * 3, rij_OO.shape[0], 3, nconfigs_test))

    features_total = np.concatenate((features_EG2OO_new, features_EG2OH_new), axis=0)

    return features_total

'''
def features_wannier_peturb(vecrij_OO, vecrij_OH, EO, EH):
    rij_OO = np.linalg.norm(vecrij_OO, axis=2)
    rij_OH = np.linalg.norm(vecrij_OH, axis=2)

    ijones_diag = np.zeros((rij_OO.shape[0], rij_OO.shape[1], 1, 1), dtype=np.float32)
    np.fill_diagonal(ijones_diag[:, :, 0, 0], 100)
    rij_OO += ijones_diag

    cosalpha_OO_xax = vecrij_OO[:, :, 0, :, :] / rij_OO
    cosalpha_OO_yax = vecrij_OO[:, :, 1, :, :] / rij_OO
    cosalpha_OO_zax = vecrij_OO[:, :, 2, :, :] / rij_OO

    cosalpha_OH_xax = vecrij_OH[:, :, 0, :, :] / rij_OH
    cosalpha_OH_yax = vecrij_OH[:, :, 1, :, :] / rij_OH
    cosalpha_OH_zax = vecrij_OH[:, :, 2, :, :] / rij_OH

    rij_OO_e = np.expand_dims(rij_OO, axis=2)
    rij_OH_e = np.expand_dims(rij_OH, axis=2)

    cosalpha_OO_xax_e = np.expand_dims(cosalpha_OO_xax, axis=2)
    cosalpha_OO_yax_e = np.expand_dims(cosalpha_OO_yax, axis=2)
    cosalpha_OO_zax_e = np.expand_dims(cosalpha_OO_zax, axis=2)

    cosalpha_OH_xax_e = np.expand_dims(cosalpha_OH_xax, axis=2)
    cosalpha_OH_yax_e = np.expand_dims(cosalpha_OH_yax, axis=2)
    cosalpha_OH_zax_e = np.expand_dims(cosalpha_OH_zax, axis=2)

    parameters2_OO = np.loadtxt("Main/G2_parameters_OO.txt")
    parameters2_OH = np.loadtxt("Main/G2_parameters_OH.txt")

    parameters4_OO = np.loadtxt("Main/G4_parameters_OO.txt")
    parameters4_OH = np.loadtxt("Main/G4_parameters_OH.txt")

    features_G2OO = np.zeros((parameters2_OO.shape[0], noxygen, 1, nfolders, nconfigs_test), dtype=np.float32)
    features_G2OH = np.zeros((parameters2_OH.shape[0], noxygen, 1, nfolders, nconfigs_test), dtype=np.float32)

    features_G4OO_zax = np.zeros((parameters4_OO.shape[0], noxygen, 1, nfolders, nconfigs_test), dtype=np.float32)
    features_G4OH_zax = np.zeros((parameters4_OH.shape[0], noxygen, 1, nfolders, nconfigs_test), dtype=np.float32)

    features_G4OO_xax = np.zeros((parameters4_OO.shape[0], noxygen, 1, nfolders, nconfigs_test), dtype=np.float32)
    features_G4OH_xax = np.zeros((parameters4_OH.shape[0], noxygen, 1, nfolders, nconfigs_test), dtype=np.float32)

    features_G4OO_yax = np.zeros((parameters4_OO.shape[0], noxygen, 1, nfolders, nconfigs_test), dtype=np.float32)
    features_G4OH_yax = np.zeros((parameters4_OH.shape[0], noxygen, 1, nfolders, nconfigs_test), dtype=np.float32)

    # features for those fields, we have additional dimension (the third one) which records the xyz of the E field
    features_EG2OO = np.zeros((parameters2_OO.shape[0], noxygen, 3, nfolders, nconfigs_test), dtype=np.float32)
    features_EG2OH = np.zeros((parameters2_OH.shape[0], noxygen, 3, nfolders, nconfigs_test), dtype=np.float32)

    features_EG4OO_zax = np.zeros((parameters4_OO.shape[0], noxygen, 3, nfolders, nconfigs_test), dtype=np.float32)
    features_EG4OH_zax = np.zeros((parameters4_OH.shape[0], noxygen, 3, nfolders, nconfigs_test), dtype=np.float32)

    features_EG4OO_xax = np.zeros((parameters4_OO.shape[0], noxygen, 3, nfolders, nconfigs_test), dtype=np.float32)
    features_EG4OH_xax = np.zeros((parameters4_OH.shape[0], noxygen, 3, nfolders, nconfigs_test), dtype=np.float32)

    features_EG4OO_yax = np.zeros((parameters4_OO.shape[0], noxygen, 3, nfolders, nconfigs_test), dtype=np.float32)
    features_EG4OH_yax = np.zeros((parameters4_OH.shape[0], noxygen, 3, nfolders, nconfigs_test), dtype=np.float32)

    for ip in range(parameters2_OO.shape[0]):
        rs = parameters2_OO[ip, 0]
        yeta = parameters2_OO[ip, 1]

        G2_ip = G2(rij_OO_e, yeta, rs)

        # get the final G20O
        G2_ip_sum = np.sum(G2_ip, axis=1)

        features_G2OO[ip, :, :, :, :] = G2_ip_sum[:, :, :, :]

    for ip in range(parameters2_OH.shape[0]):
        rs = parameters2_OH[ip, 0]
        yeta = parameters2_OH[ip, 1]

        G2_ip = G2(rij_OH_e, yeta, rs)

        # get the final G20H
        G2_ip_sum = np.sum(G2_ip, axis=1)

        features_G2OH[ip, :, :, :, :] = G2_ip_sum[:, :, :, :]

    for ip in range(parameters4_OO.shape[0]):
        rs = parameters4_OO[ip, 0]
        yeta = parameters4_OO[ip, 1]
        lam = parameters4_OO[ip, 2]
        zeta = parameters4_OO[ip, 3]

        G4_ip = G4(rij_OO_e, cosalpha_OO_zax_e, zeta, yeta, lam)

        G4_ip_sum = np.sum(G4_ip, axis=1)

        features_G4OO_zax[ip, :, :, :, :] = G4_ip_sum[:, :, :, :]

    for ip in range(parameters4_OO.shape[0]):
        rs = parameters4_OO[ip, 0]
        yeta = parameters4_OO[ip, 1]
        lam = parameters4_OO[ip, 2]
        zeta = parameters4_OO[ip, 3]

        G4_ip = G4(rij_OO_e, cosalpha_OO_xax_e, zeta, yeta, lam)

        G4_ip_sum = np.sum(G4_ip, axis=1)

        features_G4OO_xax[ip, :, :, :, :] = G4_ip_sum[:, :, :, :]

    for ip in range(parameters4_OO.shape[0]):
        rs = parameters4_OO[ip, 0]
        yeta = parameters4_OO[ip, 1]
        lam = parameters4_OO[ip, 2]
        zeta = parameters4_OO[ip, 3]

        G4_ip = G4(rij_OO_e, cosalpha_OO_yax_e, zeta, yeta, lam)

        G4_ip_sum = np.sum(G4_ip, axis=1)

        features_G4OO_yax[ip, :, :, :, :] = G4_ip_sum[:, :, :, :]

    for ip in range(parameters4_OH.shape[0]):
        rs = parameters4_OH[ip, 0]
        yeta = parameters4_OH[ip, 1]
        lam = parameters4_OH[ip, 2]
        zeta = parameters4_OH[ip, 3]

        G4_ip = G4(rij_OH_e, cosalpha_OH_zax_e, zeta, yeta, lam)

        G4_ip_sum = np.sum(G4_ip, axis=1)

        features_G4OH_zax[ip, :, :, :, :] = G4_ip_sum[:, :, :, :]

    for ip in range(parameters4_OH.shape[0]):
        rs = parameters4_OH[ip, 0]
        yeta = parameters4_OH[ip, 1]
        lam = parameters4_OH[ip, 2]
        zeta = parameters4_OH[ip, 3]

        G4_ip = G4(rij_OH_e, cosalpha_OH_xax_e, zeta, yeta, lam)

        G4_ip_sum = np.sum(G4_ip, axis=1)

        features_G4OH_xax[ip, :, :, :, :] = G4_ip_sum[:, :, :, :]

    for ip in range(parameters4_OH.shape[0]):
        rs = parameters4_OH[ip, 0]
        yeta = parameters4_OH[ip, 1]
        lam = parameters4_OH[ip, 2]
        zeta = parameters4_OH[ip, 3]

        G4_ip = G4(rij_OH_e, cosalpha_OH_yax_e, zeta, yeta, lam)

        G4_ip_sum = np.sum(G4_ip, axis=1)

        features_G4OH_yax[ip, :, :, :, :] = G4_ip_sum[:, :, :, :]

    # features for those fields
    for ip in range(parameters2_OO.shape[0]):
        rs = parameters2_OO[ip, 0]
        yeta = parameters2_OO[ip, 1]

        G2_ip = G2E(rij_OO_e, yeta, rs, EO)

        # get the final G20O
        G2_ip_sum = np.sum(G2_ip, axis=1)

        features_EG2OO[ip, :, :, :, :] = G2_ip_sum[:, :, :, :]

    for ip in range(parameters2_OH.shape[0]):
        rs = parameters2_OH[ip, 0]
        yeta = parameters2_OH[ip, 1]

        G2_ip = G2E(rij_OH_e, yeta, rs, EH)

        # get the final G20H
        G2_ip_sum = np.sum(G2_ip, axis=1)

        features_EG2OH[ip, :, :, :, :] = G2_ip_sum[:, :, :, :]

    for ip in range(parameters4_OO.shape[0]):
        rs = parameters4_OO[ip, 0]
        yeta = parameters4_OO[ip, 1]
        lam = parameters4_OO[ip, 2]
        zeta = parameters4_OO[ip, 3]

        G4_ip = G4E(rij_OO_e, cosalpha_OO_zax_e, zeta, yeta, lam, EO)

        G4_ip_sum = np.sum(G4_ip, axis=1)

        features_EG4OO_zax[ip, :, :, :, :] = G4_ip_sum[:, :, :, :]

    for ip in range(parameters4_OO.shape[0]):
        rs = parameters4_OO[ip, 0]
        yeta = parameters4_OO[ip, 1]
        lam = parameters4_OO[ip, 2]
        zeta = parameters4_OO[ip, 3]

        G4_ip = G4E(rij_OO_e, cosalpha_OO_xax_e, zeta, yeta, lam, EO)

        G4_ip_sum = np.sum(G4_ip, axis=1)

        features_EG4OO_xax[ip, :, :, :, :] = G4_ip_sum[:, :, :, :]

    for ip in range(parameters4_OO.shape[0]):
        rs = parameters4_OO[ip, 0]
        yeta = parameters4_OO[ip, 1]
        lam = parameters4_OO[ip, 2]
        zeta = parameters4_OO[ip, 3]

        G4_ip = G4E(rij_OO_e, cosalpha_OO_yax_e, zeta, yeta, lam, EO)

        G4_ip_sum = np.sum(G4_ip, axis=1)

        features_EG4OO_yax[ip, :, :, :, :] = G4_ip_sum[:, :, :, :]

    for ip in range(parameters4_OH.shape[0]):
        rs = parameters4_OH[ip, 0]
        yeta = parameters4_OH[ip, 1]
        lam = parameters4_OH[ip, 2]
        zeta = parameters4_OH[ip, 3]

        G4_ip = G4E(rij_OH_e, cosalpha_OH_zax_e, zeta, yeta, lam, EH)

        G4_ip_sum = np.sum(G4_ip, axis=1)

        features_EG4OH_zax[ip, :, :, :, :] = G4_ip_sum[:, :, :, :]

    for ip in range(parameters4_OH.shape[0]):
        rs = parameters4_OH[ip, 0]
        yeta = parameters4_OH[ip, 1]
        lam = parameters4_OH[ip, 2]
        zeta = parameters4_OH[ip, 3]

        G4_ip = G4E(rij_OH_e, cosalpha_OH_xax_e, zeta, yeta, lam, EH)

        G4_ip_sum = np.sum(G4_ip, axis=1)

        features_EG4OH_xax[ip, :, :, :, :] = G4_ip_sum[:, :, :, :]

    for ip in range(parameters4_OH.shape[0]):
        rs = parameters4_OH[ip, 0]
        yeta = parameters4_OH[ip, 1]
        lam = parameters4_OH[ip, 2]
        zeta = parameters4_OH[ip, 3]

        G4_ip = G4E(rij_OH_e, cosalpha_OH_yax_e, zeta, yeta, lam, EH)

        G4_ip_sum = np.sum(G4_ip, axis=1)

        features_EG4OH_yax[ip, :, :, :, :] = G4_ip_sum[:, :, :, :]

    features_EG2OO_new = np.transpose(features_EG2OO / features_G2OO, axes=(0, 2, 1, 3, 4)).reshape(
        (parameters2_OO.shape[0] * 3, noxygen, 3, nconfigs_test))
    features_EG2OH_new = np.transpose(features_EG2OH / features_G2OH, axes=(0, 2, 1, 3, 4)).reshape(
        (parameters2_OH.shape[0] * 3, noxygen, 3, nconfigs_test))
    features_EG4OO_H1O_new = np.transpose(features_EG4OO_zax / features_G4OO_zax, axes=(0, 2, 1, 3, 4)).reshape(
        (parameters4_OO.shape[0] * 3, noxygen, 3, nconfigs_test))
    features_EG4OO_xax_new = np.transpose(features_EG4OO_xax / features_G4OO_xax, axes=(0, 2, 1, 3, 4)).reshape(
        (parameters4_OO.shape[0] * 3, noxygen, 3, nconfigs_test))
    features_EG4OO_yax_new = np.transpose(features_EG4OO_yax / features_G4OO_yax, axes=(0, 2, 1, 3, 4)).reshape(
        (parameters4_OO.shape[0] * 3, noxygen, 3, nconfigs_test))
    features_EG4OH_H1O_new = np.transpose(features_EG4OH_zax / features_G4OH_zax, axes=(0, 2, 1, 3, 4)).reshape(
        (parameters4_OH.shape[0] * 3, noxygen, 3, nconfigs_test))
    features_EG4OH_xax_new = np.transpose(features_EG4OH_xax / features_G4OH_xax, axes=(0, 2, 1, 3, 4)).reshape(
        (parameters4_OH.shape[0] * 3, noxygen, 3, nconfigs_test))
    features_EG4OH_yax_new = np.transpose(features_EG4OH_yax / features_G4OH_yax, axes=(0, 2, 1, 3, 4)).reshape(
        (parameters4_OH.shape[0] * 3, noxygen, 3, nconfigs_test))

    features_total = np.concatenate((features_EG2OO_new, features_EG2OH_new, features_EG4OO_H1O_new,
                                     features_EG4OO_xax_new, features_EG4OO_yax_new, features_EG4OH_H1O_new,
                                     features_EG4OH_xax_new, features_EG4OH_yax_new), axis=0)

    features_av = np.expand_dims(np.mean(features_total, axis=(1, 2, 3)), axis=(1, 2, 3))
    features_std = np.expand_dims(np.std(features_total, axis=(1, 2, 3)), axis=(1, 2, 3))

    return features_total

# this is a general function that produces peturbation features for the rotated force
# it could produce features for both fO and fH
# if produce features for fO, then the name of the inputs follows the fashion in the definition of the function
# if produce features for fH, then switch O with H.

def features_force_peturb(vecrij_OO, vecrij_OH, EO, EH, parameters2_OO, parameters2_OH, parameters4_OO, parameters4_OH):
    rij_OO = np.linalg.norm(vecrij_OO, axis=2)
    rij_OH = np.linalg.norm(vecrij_OH, axis=2)

    ind_zero = rij_OO == 0
    rij_OO[ind_zero] = rij_OO[ind_zero] + 100

    cosalpha_OO_xax = vecrij_OO[:, :, 0, :, :] / rij_OO
    cosalpha_OO_yax = vecrij_OO[:, :, 1, :, :] / rij_OO
    cosalpha_OO_zax = vecrij_OO[:, :, 2, :, :] / rij_OO

    cosalpha_OH_xax = vecrij_OH[:, :, 0, :, :] / rij_OH
    cosalpha_OH_yax = vecrij_OH[:, :, 1, :, :] / rij_OH
    cosalpha_OH_zax = vecrij_OH[:, :, 2, :, :] / rij_OH

    rij_OO_e = np.expand_dims(rij_OO, axis=2)
    rij_OH_e = np.expand_dims(rij_OH, axis=2)

    cosalpha_OO_xax_e = np.expand_dims(cosalpha_OO_xax, axis=2)
    cosalpha_OO_yax_e = np.expand_dims(cosalpha_OO_yax, axis=2)
    cosalpha_OO_zax_e = np.expand_dims(cosalpha_OO_zax, axis=2)

    cosalpha_OH_xax_e = np.expand_dims(cosalpha_OH_xax, axis=2)
    cosalpha_OH_yax_e = np.expand_dims(cosalpha_OH_yax, axis=2)
    cosalpha_OH_zax_e = np.expand_dims(cosalpha_OH_zax, axis=2)

    features_G2OO = np.zeros((parameters2_OO.shape[0], rij_OO.shape[0], 1, nfolders, nconfigs_test), dtype=np.float32)
    features_G2OH = np.zeros((parameters2_OH.shape[0], rij_OO.shape[0], 1, nfolders, nconfigs_test), dtype=np.float32)

    features_G4OO_zax = np.zeros((parameters4_OO.shape[0], rij_OO.shape[0], 1, nfolders, nconfigs_test), dtype=np.float32)
    features_G4OH_zax = np.zeros((parameters4_OH.shape[0], rij_OO.shape[0], 1, nfolders, nconfigs_test), dtype=np.float32)

    features_G4OO_xax = np.zeros((parameters4_OO.shape[0], rij_OO.shape[0], 1, nfolders, nconfigs_test), dtype=np.float32)
    features_G4OH_xax = np.zeros((parameters4_OH.shape[0], rij_OO.shape[0], 1, nfolders, nconfigs_test), dtype=np.float32)

    features_G4OO_yax = np.zeros((parameters4_OO.shape[0], rij_OO.shape[0], 1, nfolders, nconfigs_test), dtype=np.float32)
    features_G4OH_yax = np.zeros((parameters4_OH.shape[0], rij_OO.shape[0], 1, nfolders, nconfigs_test), dtype=np.float32)

    # features for those fields, we have additional dimension (the third one) which records the xyz of the E field
    features_EG2OO = np.zeros((parameters2_OO.shape[0], rij_OO.shape[0], 3, nfolders, nconfigs_test), dtype=np.float32)
    features_EG2OH = np.zeros((parameters2_OH.shape[0], rij_OO.shape[0], 3, nfolders, nconfigs_test), dtype=np.float32)

    features_EG4OO_zax = np.zeros((parameters4_OO.shape[0], rij_OO.shape[0], 3, nfolders, nconfigs_test), dtype=np.float32)
    features_EG4OH_zax = np.zeros((parameters4_OH.shape[0], rij_OO.shape[0], 3, nfolders, nconfigs_test), dtype=np.float32)

    features_EG4OO_xax = np.zeros((parameters4_OO.shape[0], rij_OO.shape[0], 3, nfolders, nconfigs_test), dtype=np.float32)
    features_EG4OH_xax = np.zeros((parameters4_OH.shape[0], rij_OO.shape[0], 3, nfolders, nconfigs_test), dtype=np.float32)

    features_EG4OO_yax = np.zeros((parameters4_OO.shape[0], rij_OO.shape[0], 3, nfolders, nconfigs_test), dtype=np.float32)
    features_EG4OH_yax = np.zeros((parameters4_OH.shape[0], rij_OO.shape[0], 3, nfolders, nconfigs_test), dtype=np.float32)

    for ip in range(parameters2_OO.shape[0]):
        rs = parameters2_OO[ip, 0]
        yeta = parameters2_OO[ip, 1]

        G2_ip = G2(rij_OO_e, yeta, rs)

        # get the final G20O
        G2_ip_sum = np.sum(G2_ip, axis=1)

        features_G2OO[ip, :, :, :, :] = G2_ip_sum[:, :, :, :]

    for ip in range(parameters2_OH.shape[0]):
        rs = parameters2_OH[ip, 0]
        yeta = parameters2_OH[ip, 1]

        G2_ip = G2(rij_OH_e, yeta, rs)

        # get the final G20H
        G2_ip_sum = np.sum(G2_ip, axis=1)

        features_G2OH[ip, :, :, :, :] = G2_ip_sum[:, :, :, :]

    for ip in range(parameters4_OO.shape[0]):
        rs = parameters4_OO[ip, 0]
        yeta = parameters4_OO[ip, 1]
        lam = parameters4_OO[ip, 2]
        zeta = parameters4_OO[ip, 3]

        G4_ip = G4(rij_OO_e, cosalpha_OO_zax_e, zeta, yeta, lam)

        G4_ip_sum = np.sum(G4_ip, axis=1)

        features_G4OO_zax[ip, :, :, :, :] = G4_ip_sum[:, :, :, :]

    for ip in range(parameters4_OO.shape[0]):
        rs = parameters4_OO[ip, 0]
        yeta = parameters4_OO[ip, 1]
        lam = parameters4_OO[ip, 2]
        zeta = parameters4_OO[ip, 3]

        G4_ip = G4(rij_OO_e, cosalpha_OO_xax_e, zeta, yeta, lam)

        G4_ip_sum = np.sum(G4_ip, axis=1)

        features_G4OO_xax[ip, :, :, :, :] = G4_ip_sum[:, :, :, :]

    for ip in range(parameters4_OO.shape[0]):
        rs = parameters4_OO[ip, 0]
        yeta = parameters4_OO[ip, 1]
        lam = parameters4_OO[ip, 2]
        zeta = parameters4_OO[ip, 3]

        G4_ip = G4(rij_OO_e, cosalpha_OO_yax_e, zeta, yeta, lam)

        G4_ip_sum = np.sum(G4_ip, axis=1)

        features_G4OO_yax[ip, :, :, :, :] = G4_ip_sum[:, :, :, :]

    for ip in range(parameters4_OH.shape[0]):
        rs = parameters4_OH[ip, 0]
        yeta = parameters4_OH[ip, 1]
        lam = parameters4_OH[ip, 2]
        zeta = parameters4_OH[ip, 3]

        G4_ip = G4(rij_OH_e, cosalpha_OH_zax_e, zeta, yeta, lam)

        G4_ip_sum = np.sum(G4_ip, axis=1)

        features_G4OH_zax[ip, :, :, :, :] = G4_ip_sum[:, :, :, :]

    for ip in range(parameters4_OH.shape[0]):
        rs = parameters4_OH[ip, 0]
        yeta = parameters4_OH[ip, 1]
        lam = parameters4_OH[ip, 2]
        zeta = parameters4_OH[ip, 3]

        G4_ip = G4(rij_OH_e, cosalpha_OH_xax_e, zeta, yeta, lam)

        G4_ip_sum = np.sum(G4_ip, axis=1)

        features_G4OH_xax[ip, :, :, :, :] = G4_ip_sum[:, :, :, :]

    for ip in range(parameters4_OH.shape[0]):
        rs = parameters4_OH[ip, 0]
        yeta = parameters4_OH[ip, 1]
        lam = parameters4_OH[ip, 2]
        zeta = parameters4_OH[ip, 3]

        G4_ip = G4(rij_OH_e, cosalpha_OH_yax_e, zeta, yeta, lam)

        G4_ip_sum = np.sum(G4_ip, axis=1)

        features_G4OH_yax[ip, :, :, :, :] = G4_ip_sum[:, :, :, :]

    # features for those fields
    for ip in range(parameters2_OO.shape[0]):
        rs = parameters2_OO[ip, 0]
        yeta = parameters2_OO[ip, 1]

        G2_ip = G2E(rij_OO_e, yeta, rs, EO)

        # get the final G20O
        G2_ip_sum = np.sum(G2_ip, axis=1)

        features_EG2OO[ip, :, :, :, :] = G2_ip_sum[:, :, :, :]

    for ip in range(parameters2_OH.shape[0]):
        rs = parameters2_OH[ip, 0]
        yeta = parameters2_OH[ip, 1]

        G2_ip = G2E(rij_OH_e, yeta, rs, EH)

        # get the final G20H
        G2_ip_sum = np.sum(G2_ip, axis=1)

        features_EG2OH[ip, :, :, :, :] = G2_ip_sum[:, :, :, :]

    for ip in range(parameters4_OO.shape[0]):
        rs = parameters4_OO[ip, 0]
        yeta = parameters4_OO[ip, 1]
        lam = parameters4_OO[ip, 2]
        zeta = parameters4_OO[ip, 3]

        G4_ip = G4E(rij_OO_e, cosalpha_OO_zax_e, zeta, yeta, lam, EO)

        G4_ip_sum = np.sum(G4_ip, axis=1)

        features_EG4OO_zax[ip, :, :, :, :] = G4_ip_sum[:, :, :, :]

    for ip in range(parameters4_OO.shape[0]):
        rs = parameters4_OO[ip, 0]
        yeta = parameters4_OO[ip, 1]
        lam = parameters4_OO[ip, 2]
        zeta = parameters4_OO[ip, 3]

        G4_ip = G4E(rij_OO_e, cosalpha_OO_xax_e, zeta, yeta, lam, EO)

        G4_ip_sum = np.sum(G4_ip, axis=1)

        features_EG4OO_xax[ip, :, :, :, :] = G4_ip_sum[:, :, :, :]

    for ip in range(parameters4_OO.shape[0]):
        rs = parameters4_OO[ip, 0]
        yeta = parameters4_OO[ip, 1]
        lam = parameters4_OO[ip, 2]
        zeta = parameters4_OO[ip, 3]

        G4_ip = G4E(rij_OO_e, cosalpha_OO_yax_e, zeta, yeta, lam, EO)

        G4_ip_sum = np.sum(G4_ip, axis=1)

        features_EG4OO_yax[ip, :, :, :, :] = G4_ip_sum[:, :, :, :]

    for ip in range(parameters4_OH.shape[0]):
        rs = parameters4_OH[ip, 0]
        yeta = parameters4_OH[ip, 1]
        lam = parameters4_OH[ip, 2]
        zeta = parameters4_OH[ip, 3]

        G4_ip = G4E(rij_OH_e, cosalpha_OH_zax_e, zeta, yeta, lam, EH)

        G4_ip_sum = np.sum(G4_ip, axis=1)

        features_EG4OH_zax[ip, :, :, :, :] = G4_ip_sum[:, :, :, :]

    for ip in range(parameters4_OH.shape[0]):
        rs = parameters4_OH[ip, 0]
        yeta = parameters4_OH[ip, 1]
        lam = parameters4_OH[ip, 2]
        zeta = parameters4_OH[ip, 3]

        G4_ip = G4E(rij_OH_e, cosalpha_OH_xax_e, zeta, yeta, lam, EH)

        G4_ip_sum = np.sum(G4_ip, axis=1)

        features_EG4OH_xax[ip, :, :, :, :] = G4_ip_sum[:, :, :, :]

    for ip in range(parameters4_OH.shape[0]):
        rs = parameters4_OH[ip, 0]
        yeta = parameters4_OH[ip, 1]
        lam = parameters4_OH[ip, 2]
        zeta = parameters4_OH[ip, 3]

        G4_ip = G4E(rij_OH_e, cosalpha_OH_yax_e, zeta, yeta, lam, EH)

        G4_ip_sum = np.sum(G4_ip, axis=1)

        features_EG4OH_yax[ip, :, :, :, :] = G4_ip_sum[:, :, :, :]

    features_EG2OO_new = np.transpose(features_EG2OO / features_G2OO, axes=(0, 2, 1, 3, 4)).reshape(
        (parameters2_OO.shape[0] * 3, rij_OO.shape[0], 3, nconfigs_test))
    features_EG2OH_new = np.transpose(features_EG2OH / features_G2OH, axes=(0, 2, 1, 3, 4)).reshape(
        (parameters2_OH.shape[0] * 3, rij_OO.shape[0], 3, nconfigs_test))
    features_EG4OO_H1O_new = np.transpose(features_EG4OO_zax / features_G4OO_zax, axes=(0, 2, 1, 3, 4)).reshape(
        (parameters4_OO.shape[0] * 3, rij_OO.shape[0], 3, nconfigs_test))
    features_EG4OO_xax_new = np.transpose(features_EG4OO_xax / features_G4OO_xax, axes=(0, 2, 1, 3, 4)).reshape(
        (parameters4_OO.shape[0] * 3, rij_OO.shape[0], 3, nconfigs_test))
    features_EG4OO_yax_new = np.transpose(features_EG4OO_yax / features_G4OO_yax, axes=(0, 2, 1, 3, 4)).reshape(
        (parameters4_OO.shape[0] * 3, rij_OO.shape[0], 3, nconfigs_test))
    features_EG4OH_H1O_new = np.transpose(features_EG4OH_zax / features_G4OH_zax, axes=(0, 2, 1, 3, 4)).reshape(
        (parameters4_OH.shape[0] * 3, rij_OO.shape[0], 3, nconfigs_test))
    features_EG4OH_xax_new = np.transpose(features_EG4OH_xax / features_G4OH_xax, axes=(0, 2, 1, 3, 4)).reshape(
        (parameters4_OH.shape[0] * 3, rij_OO.shape[0], 3, nconfigs_test))
    features_EG4OH_yax_new = np.transpose(features_EG4OH_yax / features_G4OH_yax, axes=(0, 2, 1, 3, 4)).reshape(
        (parameters4_OH.shape[0] * 3, rij_OO.shape[0], 3, nconfigs_test))

    features_total = np.concatenate((features_EG2OO_new, features_EG2OH_new, features_EG4OO_H1O_new,
                                     features_EG4OO_xax_new, features_EG4OO_yax_new, features_EG4OH_H1O_new,
                                     features_EG4OH_xax_new, features_EG4OH_yax_new), axis=0)

    features_av = np.expand_dims(np.mean(features_total, axis=(1, 2, 3)), axis=(1, 2, 3))
    features_std = np.expand_dims(np.std(features_total, axis=(1, 2, 3)), axis=(1, 2, 3))

    return features_total
'''
