import numpy as np
from parameters import *

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

    EO_all_reshape = EO_all_stack.reshape((noxygen, 3, nfolders, nconfigs))
    EH_all_reshape = EH_all_stack.reshape((nhydrogen, 3, nfolders, nconfigs))
    Ew_all_reshape = Ew_all_stack.reshape((nwannier, 3, nfolders, nconfigs))

    Eexternal = np.array([0, 0, 0, 0, 0, 0, 0, 0.1/51.4, 0.2/51.4]).reshape((3, nfolders, 1))  # the field is applied to the z-direction, 51.4 is the factor that converts the field from V/A to atomic unit

    EO_sum = EO_all_reshape + Eexternal
    EH_sum = EH_all_reshape + Eexternal
    Ew_sum = Ew_all_reshape + Eexternal

    return EO_sum, EH_sum, Ew_sum
