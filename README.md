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
 
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.9.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.9.0%2Bcpu.zip 

4. Compile the code
I will take the Liquid Vapor interface simulation as an example to show you how to compile the code.
First go to examples/LiquidVapor:
cd examples/LiquidVapor

Then at the 16th line of CMakeLists.txt, change /dssg/home/gaoang/libtorch to the directory where your install libtorch.

Then go to demo/

cd demo

Then enter the singularity container:

singularity shell -e /path-to-inte_oneapikit/oneapi-hpckit_latest.sif

Then run the run_make.sh:
./run_make.sh

Now the compilation will begin. It might takes several minutes to compile the code, depending on your hardware. The compiled executable is named MD.

Instructions for run SCFNN

Still let's use the LiquidVapor interface as an example. Suppose you are still in the folder examples/LiquidVapor/demo, where you have just compiled the SCFNN. 

First, you need to enter the singularity container again:

singularity shell -e /path-to-inte_oneapikit/oneapi-hpckit_latest.sif

Now run the code with
./run.sh

run.sh contains parameters such as simulation steps, timestep, thermostat ect. One can go to the main_final_twobath.cpp. The first lines of in the main() function explains the parameters.

Notice that the simulation requires Oxyz.txt, Hxyz.txt and box.txt as inputs. Oxyz.txt and Hxyz.txt contains the intial coordinates. The box.txt contains the box dimension.

Sample output files can be found in demo/. Among the output files, the most important ones are xyz_hist.txt and wxyz_hist.txt, which are the trajectory files of the coordinates of the nuclues and the wannier centers respectively. 

It may take several minutes to an hour to complete the demo simulation, depending on the hardware you have. Notice that the output you got might be different from the sample outputs provided. This is because random number generator is used in the code, thus each run will generate different results.
Should you have any questions, please contact anggao@bupt.edu.cn
