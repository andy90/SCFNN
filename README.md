# SCFNN
This folder contains the codes for the self consistent field neural network (SCFNN) model. 

The train_test/ folder contains the files used to train and test the SCFNN model. The train and dest data files is kept in https://doi.org/10.5281/zenodo.5521328. One should download the data files separately, and move the three folders, D0/, D0p1V and D0p2V, to the train_test/ folder. After that we can run the codes to train and test the model.

The MDcode/ folder contains the molecular dynamics simulation code to run the water simulation using SCFNN. It also contains the MD code to run the water simulation using the Behler-Parrinello model. It also cotains the code to do the finite-field constant D simulation using SCFNN. 

In examples/, we provide sample inputs and outputs for the LiquidVapor Interface simulation and the Kirkwood g-factor calculation.

Installation Guide

Here we provide a step-to-step installation guide for the SCFNN based on the container technology. Container is a light-weight virturalization technology that allows your code to be run on any platform (including Linux, Windows and MacOS). Here we will use Singularity container (https://sylabs.io) as example. The following installating steps have been tested on CentOS 7.

1. Install Singularity
The installation guide for singularity can be found on https://sylabs.io/guides/3.0/user-guide/installation.html#install-on-windows-or-mac. The version of singularity we used in this example is 3.1.0.

2. Pull the intel hpc image
To compile the SCFNN code, we need to use the intel compiler. Pull the intel oneapi hpc kit using the following command:
singularity pull oneapi-hpckit_latest.sif docker://intel/oneapi-hpckit 

3. Download libtorch
The SCFNN code utilizes libtorch to do the neural network related calculations. We need to download libtorch in order to compile SCFNN. The libtorch version we used is 1.9.0.
wget 
https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.9.0%2Bcpu.zip
 
Should you have any questions, please contact anggao@bupt.edu.cn
