import numpy as np
from parameters import *
import torch
from torch import nn
import torch.optim as optim
from sklearn.decomposition import PCA
from useful_functions import *

def train_wannier_peturb(wannierxyz, features):
    npca = 5
    x = torch.tensor(np.transpose(features, axes=(2, 3, 1, 0)), dtype=torch.float)


    y_0 = np.transpose(wannierxyz, axes=(3, 4, 0, 1, 2)).reshape(wannierxyz.shape[3], wannierxyz.shape[4],
                                                                 wannierxyz.shape[0],
                                                                 wannierxyz.shape[1] * wannierxyz.shape[2])
    y = torch.tensor(y_0, dtype=torch.float)


    class WCNet(nn.Module):
        def __init__(self):
            super(WCNet, self).__init__()
            n_first = features.shape[0]
            n_second = 12
            self.linear_stack = nn.Sequential(
                nn.Linear(n_first, n_second, bias=False),  # setting the bias equal 0 can make sure the
            )

        def forward(self, x):
            y = self.linear_stack(x)
            return y

    net = WCNet()

    optimizer = optim.Adam(net.parameters())
    for iepoch in range(1000):
        optimizer.zero_grad()  # zero the gradient buffers
        y_pred = net(x)
        loss1 = torch.sum(torch.abs(y_pred[1] - y_pred[0] - (y[1] - y[0])))
        loss2 = torch.sum(torch.abs(y_pred[2] - y_pred[0] - (y[2] - y[0])))
        loss3 = torch.sum(torch.abs(y_pred[2] - y_pred[1] - (y[2] - y[1])))
        loss = loss1 + loss2 + loss3
        loss.backward()
        optimizer.step()
        print(iepoch, torch.median(torch.abs(y_pred[2] - y_pred[0] - (y[2] - y[0]))))
    print(torch.median(torch.abs(y[2] - y[0])))

    y_pred_final = net(x)
    y_pred_reshaped = backward_axis(uncompress_dims(y_pred_final.detach().numpy(), 3, 4))

    torch.save(net.state_dict(), "wannier_peturb.pth")
    net_traced = torch.jit.trace(net, x)
    net_traced.save("traced_wannier_peturb.pt")

    return y_pred_reshaped

def train_wannier_GT(wannierxyz_GT, features):
    x = torch.tensor(np.transpose(features, axes=(2, 1, 0)), dtype=torch.float)

    y_1 = torch.tensor(compress_dims(np.transpose(wannierxyz_GT, axes=(3, 0, 1, 2)), 2), dtype=torch.float)

    y_1_av = torch.mean(y_1, axis=(0, 1))  # the average for each xyz coordinate of the wannier center

    y = (y_1 - y_1_av)  # scale the xyz coordinate of the wannier center

    class WCNet(nn.Module):
        def __init__(self):
            super(WCNet, self).__init__()
            n_first = 36
            n_second = 24
            n_third = 16
            n_forth = 12
            self.linear_tanh_stack = nn.Sequential(
                nn.Linear(n_first, n_second),
                nn.Tanh(),
                nn.Linear(n_second, n_third),
                nn.Tanh(),
                nn.Linear(n_third, n_forth),
            )

        def forward(self, x):
            y = self.linear_tanh_stack(x)
            return y

    net = WCNet()

    optimizer = optim.Adam(net.parameters())

    for iepoch in range(500):
        for i in range(x.shape[0]):
            optimizer.zero_grad()  # zero the gradient buffers
            y_pred = net(x[i])
            loss = torch.sum(torch.abs(y_pred - y[i]))
            loss.backward()
            optimizer.step()
            if (iepoch % 20 == 0) & (i % 20 == 0):
                print(iepoch, loss)

    y_pred_final = net(x)

    print(torch.mean(torch.abs(y - y_pred_final), axis=(0, 1)))  # the average error for each xyz of the wannier center
    for i in range(12):
        print(torch.median(torch.abs(y - y_pred_final)[:, :, i]))

    y_pred_descale = y_pred_final + y_1_av
    wannierxyz_GT_predict = np.transpose(uncompress_dims(y_pred_descale.detach().numpy(), 2, 4), axes=(1, 2, 3, 0))

    torch.save(net.state_dict(), "wannier_GT.pth")
    net_traced = torch.jit.trace(net, x)
    net_traced.save("traced_wannier_GT.pt")

    np.savetxt("wannier_GT_target_scale.txt", y_1_av)

    return wannierxyz_GT_predict

def train_force_peturb(fO, features, mol_type):
    
    x = torch.tensor(np.transpose(features, axes=(2, 3, 1, 0)), dtype=torch.float)
    y_0 = forward_axis(fO)
    y = torch.tensor(y_0, dtype=torch.float)

    class WCNet(nn.Module):
        def __init__(self):
            super(WCNet, self).__init__()
            n_first = features.shape[0]
            n_second = 3
            self.linear_stack = nn.Sequential(
                nn.Linear(n_first, n_second, bias=False),  # setting the bias equal 0 can make sure the
            )

        def forward(self, x):
            y = self.linear_stack(x)
            return y

    net = WCNet()

    optimizer = optim.Adam(net.parameters())
    # losscombine = torch.nn.SmoothL1Loss(reduction='mean', beta=0.001)
    for iepoch in range(2000):
        optimizer.zero_grad()  # zero the gradient buffers
        y_pred = net(x)
        # loss = torch.sum(torch.abs(y_pred - y))
        # loss = losscombine(y_pred, y)
        loss1 = torch.sum(torch.abs(y_pred[1] - y_pred[0] - (y[1] - y[0])))
        loss2 = torch.sum(torch.abs(y_pred[2] - y_pred[0] - (y[2] - y[0])))
        loss3 = torch.sum(torch.abs(y_pred[2] - y_pred[1] - (y[2] - y[1])))
        loss = loss1 + loss2 + loss3
        loss.backward()
        optimizer.step()
        print(iepoch, torch.median(torch.abs(y_pred[2] - y_pred[0] - (y[2] - y[0]))))
    print(torch.median(torch.abs(y[2] - y[0])))
    y_pred_final = net(x)

    torch.save(net.state_dict(), "force_peturb_" + mol_type + ".pth")
    net_traced = torch.jit.trace(net, x)
    net_traced.save("force_peturb_" + mol_type + ".pt")

    return backward_axis(y_pred_final.detach().numpy())


def train_force_GT(fO, fH):
    xO = np.load("xO.npy")
    xH = np.load("xH.npy")
    xOO_d = np.load("xOO_d.npy")
    xOH_d = np.load("xOH_d.npy")
    xHO_d = np.load("xHO_d.npy")
    xHH_d = np.load("xHH_d.npy")

    xO = torch.tensor(xO, dtype=torch.float)
    xH = torch.tensor(xH, dtype=torch.float)
    xOO_d = torch.tensor(xOO_d, dtype=torch.float)
    xOH_d = torch.tensor(xOH_d, dtype=torch.float)
    xHO_d = torch.tensor(xHO_d, dtype=torch.float)
    xHH_d = torch.tensor(xHH_d, dtype=torch.float)

    fO = np.transpose(fO, axes=(2, 0, 1)) # move the config axis to the front
    fH = np.transpose(fH, axes=(2, 0, 1))

    yO = torch.tensor(fO, dtype=torch.float) / 0.05  # make the standard deviation of the forces to be about 1
    yH = torch.tensor(fH, dtype=torch.float) / 0.05

    class BPNet(nn.Module):
        def __init__(self):
            super(BPNet, self).__init__()
            n_first_O = 30
            n_second_O = 25
            n_third_O = 25
            self.w1_O = nn.Parameter(torch.randn((n_first_O, n_second_O))/5)
            self.b1_O = nn.Parameter(torch.randn(n_second_O)/5)
            self.w2_O = nn.Parameter(torch.randn((n_second_O, n_third_O))/5)
            self.b2_O = nn.Parameter(torch.randn(n_third_O)/5)
            self.w3_O = nn.Parameter(torch.randn((n_third_O, 1))/5)
            self.b3_O = nn.Parameter(torch.randn(1)/5)

            n_first_H = 27
            n_second_H = 25
            n_third_H = 25
            self.w1_H = nn.Parameter(torch.randn((n_first_H, n_second_H)) / 5)
            self.b1_H = nn.Parameter(torch.randn(n_second_H) / 5)
            self.w2_H = nn.Parameter(torch.randn((n_second_H, n_third_H)) / 5)
            self.b2_H = nn.Parameter(torch.randn(n_third_H) / 5)
            self.w3_H = nn.Parameter(torch.randn((n_third_H, 1)) / 5)
            self.b3_H = nn.Parameter(torch.randn(1) / 5)

        def forward(self, x_O, x_H, dx_OO, dx_HO, dx_OH, dx_HH):
            z1_O = torch.matmul(x_O, self.w1_O) + self.b1_O
            z2_O = torch.matmul(torch.tanh(z1_O), self.w2_O) + self.b2_O

            z1_H = torch.matmul(x_H, self.w1_H) + self.b1_H
            z2_H = torch.matmul(torch.tanh(z1_H), self.w2_H) + self.b2_H

            ap1_OO = torch.matmul(dx_OO, self.w1_O) / torch.cosh(z1_O) ** 2
            ap2_OO = torch.matmul(ap1_OO, self.w2_O) / torch.cosh(z2_O) ** 2
            y_OO = torch.matmul(ap2_OO, self.w3_O)

            ap1_HO = torch.matmul(dx_HO, self.w1_O) / torch.cosh(z1_O) ** 2
            ap2_HO = torch.matmul(ap1_HO, self.w2_O) / torch.cosh(z2_O) ** 2
            y_HO = torch.matmul(ap2_HO, self.w3_O)

            ap1_HH = torch.matmul(dx_HH, self.w1_H) / torch.cosh(z1_H) ** 2
            ap2_HH = torch.matmul(ap1_HH, self.w2_H) / torch.cosh(z2_H) ** 2
            y_HH = torch.matmul(ap2_HH, self.w3_H)

            ap1_OH = torch.matmul(dx_OH, self.w1_H) / torch.cosh(z1_H) ** 2
            ap2_OH = torch.matmul(ap1_OH, self.w2_H) / torch.cosh(z2_H) ** 2
            y_OH = torch.matmul(ap2_OH, self.w3_H)

            y_O = torch.sum(y_OO, axis=(-1, -2)) + torch.sum(y_OH, axis=(-1, -2))
            y_H = torch.sum(y_HO, axis=(-1, -2)) + torch.sum(y_HH, axis=(-1, -2))  # this is like the change of total energy resulted by the change of H
            return y_O, y_H

    net = BPNet()

    optimizer = optim.Adam(net.parameters())

    for iepoch in range(200):
        for i in range(xO.shape[0]):
            optimizer.zero_grad()  # zero the gradient buffers
            yO_pred, yH_pred = net(xO[i], xH[i], xOO_d[i], xHO_d[i], xOH_d[i], xHH_d[i])
            loss = torch.sum(torch.abs(yO_pred - yO[i])) + torch.sum(torch.abs(yH_pred - yH[i]))
            loss.backward()
            optimizer.step()
            if (iepoch % 20 == 0) & (i % 20 == 0):
                print(iepoch, loss)

    yO_pred_all = ()
    yH_pred_all = ()
    for i in range(xO.shape[0]):
        yO_pred, yH_pred = net(xO[i], xH[i], xOO_d[i], xHO_d[i], xOH_d[i], xHH_d[i])
        yO_pred_all += (yO_pred, )
        yH_pred_all += (yH_pred, )

    yO_pred_stack = torch.stack(yO_pred_all, -1).detach().numpy()
    yH_pred_stack = torch.stack(yH_pred_all, -1).detach().numpy()

    torch.save(net.state_dict(), "trained_force_model_statedict.pth")

    net_traced = torch.jit.trace(net, (xO[0], xH[0], xOO_d[0], xHO_d[0], xOH_d[0], xHH_d[0]))
    net_traced.save("traced_trained_model.pt")

    return yO_pred_stack, yH_pred_stack

def train_force_BP(fO, fH):
    xO = np.load("xO.npy")
    xH = np.load("xH.npy")
    xOO_d = np.load("xOO_d.npy")
    xOH_d = np.load("xOH_d.npy")
    xHO_d = np.load("xHO_d.npy")
    xHH_d = np.load("xHH_d.npy")

    xO = torch.tensor(xO, dtype=torch.float)
    xH = torch.tensor(xH, dtype=torch.float)
    xOO_d = torch.tensor(xOO_d, dtype=torch.float)
    xOH_d = torch.tensor(xOH_d, dtype=torch.float)
    xHO_d = torch.tensor(xHO_d, dtype=torch.float)
    xHH_d = torch.tensor(xHH_d, dtype=torch.float)

    fO = np.transpose(fO, axes=(2, 0, 1)) # move the config axis to the front
    fH = np.transpose(fH, axes=(2, 0, 1))

    yO = torch.tensor(fO, dtype=torch.float) / 0.05  # make the standard deviation of the forces to be about 1
    yH = torch.tensor(fH, dtype=torch.float) / 0.05

    class BPNet(nn.Module):
        def __init__(self):
            super(BPNet, self).__init__()
            n_first_O = 30
            n_second_O = 25
            n_third_O = 25
            self.w1_O = nn.Parameter(torch.randn((n_first_O, n_second_O))/5)
            self.b1_O = nn.Parameter(torch.randn(n_second_O)/5)
            self.w2_O = nn.Parameter(torch.randn((n_second_O, n_third_O))/5)
            self.b2_O = nn.Parameter(torch.randn(n_third_O)/5)
            self.w3_O = nn.Parameter(torch.randn((n_third_O, 1))/5)
            self.b3_O = nn.Parameter(torch.randn(1)/5)

            n_first_H = 27
            n_second_H = 25
            n_third_H = 25
            self.w1_H = nn.Parameter(torch.randn((n_first_H, n_second_H)) / 5)
            self.b1_H = nn.Parameter(torch.randn(n_second_H) / 5)
            self.w2_H = nn.Parameter(torch.randn((n_second_H, n_third_H)) / 5)
            self.b2_H = nn.Parameter(torch.randn(n_third_H) / 5)
            self.w3_H = nn.Parameter(torch.randn((n_third_H, 1)) / 5)
            self.b3_H = nn.Parameter(torch.randn(1) / 5)

        def forward(self, x_O, x_H, dx_OO, dx_HO, dx_OH, dx_HH):
            z1_O = torch.matmul(x_O, self.w1_O) + self.b1_O
            z2_O = torch.matmul(torch.tanh(z1_O), self.w2_O) + self.b2_O

            z1_H = torch.matmul(x_H, self.w1_H) + self.b1_H
            z2_H = torch.matmul(torch.tanh(z1_H), self.w2_H) + self.b2_H

            ap1_OO = torch.matmul(dx_OO, self.w1_O) / torch.cosh(z1_O) ** 2
            ap2_OO = torch.matmul(ap1_OO, self.w2_O) / torch.cosh(z2_O) ** 2
            y_OO = torch.matmul(ap2_OO, self.w3_O)

            ap1_HO = torch.matmul(dx_HO, self.w1_O) / torch.cosh(z1_O) ** 2
            ap2_HO = torch.matmul(ap1_HO, self.w2_O) / torch.cosh(z2_O) ** 2
            y_HO = torch.matmul(ap2_HO, self.w3_O)

            ap1_HH = torch.matmul(dx_HH, self.w1_H) / torch.cosh(z1_H) ** 2
            ap2_HH = torch.matmul(ap1_HH, self.w2_H) / torch.cosh(z2_H) ** 2
            y_HH = torch.matmul(ap2_HH, self.w3_H)

            ap1_OH = torch.matmul(dx_OH, self.w1_H) / torch.cosh(z1_H) ** 2
            ap2_OH = torch.matmul(ap1_OH, self.w2_H) / torch.cosh(z2_H) ** 2
            y_OH = torch.matmul(ap2_OH, self.w3_H)

            y_O = torch.sum(y_OO, axis=(-1, -2)) + torch.sum(y_OH, axis=(-1, -2))
            y_H = torch.sum(y_HO, axis=(-1, -2)) + torch.sum(y_HH, axis=(-1, -2))  # this is like the change of total energy resulted by the change of H
            return y_O, y_H

    net = BPNet()

    optimizer = optim.Adam(net.parameters())

    for iepoch in range(200):
        for i in range(xO.shape[0]):
            optimizer.zero_grad()  # zero the gradient buffers
            yO_pred, yH_pred = net(xO[i], xH[i], xOO_d[i], xHO_d[i], xOH_d[i], xHH_d[i])
            loss = torch.sum(torch.abs(yO_pred - yO[i])) + torch.sum(torch.abs(yH_pred - yH[i]))
            loss.backward()
            optimizer.step()
            if (iepoch % 20 == 0) & (i % 20 == 0):
                print(iepoch, loss)

    yO_pred_all = ()
    yH_pred_all = ()
    for i in range(xO.shape[0]):
        yO_pred, yH_pred = net(xO[i], xH[i], xOO_d[i], xHO_d[i], xOH_d[i], xHH_d[i])
        yO_pred_all += (yO_pred, )
        yH_pred_all += (yH_pred, )

    yO_pred_stack = torch.stack(yO_pred_all, -1).detach().numpy()
    yH_pred_stack = torch.stack(yH_pred_all, -1).detach().numpy()

    torch.save(net.state_dict(), "trained_force_model_statedict_BP.pth")

    net_traced = torch.jit.trace(net, (xO[0], xH[0], xOO_d[0], xHO_d[0], xOH_d[0], xHH_d[0]))
    net_traced.save("traced_trained_model_BP.pt")

    return yO_pred_stack, yH_pred_stack

# this does not seem to help at all
# maybe what we need is more new features
def train_wannier_peturb_nonlinear(wannierxyz, features):
    npca = 5
    x_old = np.transpose(features, axes=(2, 3, 1, 0)).reshape(features.shape[2] * features.shape[3] * features.shape[1], features.shape[0])  # move the features axis to the last
    x_pca = torch.tensor(PCA(n_components=npca).fit(x_old).transform(x_old), dtype=torch.float)
    pca = PCA(n_components=npca).fit(x_old)

    x = x_pca.reshape(features.shape[2], features.shape[3], features.shape[1], npca)

    y_0 = np.transpose(wannierxyz, axes=(3, 4, 0, 1, 2)).reshape(wannierxyz.shape[3], wannierxyz.shape[4],
                                                                 wannierxyz.shape[0],
                                                                 wannierxyz.shape[1] * wannierxyz.shape[2])
    y = torch.tensor(y_0, dtype=torch.float)

    xo = torch.zeros((x.shape[1], x.shape[2], x.shape[3]), dtype=torch.float)

    print(x.shape, y.shape, xo.shape)

    class WCNet(nn.Module):
        def __init__(self):
            super(WCNet, self).__init__()
            n_first = npca
            n_second = 8
            n_third = 12
            n_forth = 12
            self.linear_tanh_stack = nn.Sequential(
                nn.Linear(n_first, n_second),
                nn.Tanh(),
                nn.Linear(n_second, n_third),
                nn.Tanh(),
                nn.Linear(n_third, n_forth),
            )

        def forward(self, x):
            y = self.linear_tanh_stack(x)
            return y


    net = WCNet()

    optimizer = optim.Adam(net.parameters())
    for iepoch in range(20000):
        #for i in range(x.shape[1]):
        optimizer.zero_grad()  # zero the gradient buffers
        y_pred = net(x)
        yo_pred = net(xo) # if all the features is zero, the prediction should also be zero
        # loss = torch.sum(torch.abs(y_pred - y))
        # loss = losscombine(y_pred, y)
        loss1 = torch.sum(torch.abs(y_pred[1] - y_pred[0] - (y[1] - y[0])))
        loss2 = torch.sum(torch.abs(y_pred[2] - y_pred[0] - (y[2] - y[0])))
        loss3 = torch.sum(torch.abs(y_pred[2] - y_pred[1] - (y[2] - y[1])))
        loss4 = torch.sum(torch.abs(yo_pred))
        loss = loss1 + loss2 + loss3 + loss4
        loss.backward()
        optimizer.step()
        if (iepoch % 20 == 0):
            print(iepoch, torch.median(torch.abs(y_pred[2] - y_pred[0] - (y[2] - y[0]))))
    print(torch.median(torch.abs(y[2] - y[0])))
    y_pred_final = net(x)
    y_pred_reshaped = backward_axis(uncompress_dims(y_pred_final.detach().numpy(), 3, 4))

    return y_pred_reshaped
