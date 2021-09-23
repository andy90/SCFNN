# SCFNN
This folder contains the codes for the self consistent field neural network (SCFNN) model. 

The train_test/ folder contains the files used to train and test the SCFNN model. The train and dest data files is kept in https://doi.org/10.5281/zenodo.5521328. One should download the data files separately, and move the three folders, D0/, D0p1V and D0p2V, to the train_test/ folder. After that we can run the codes to train and test the model.

The MDcode/ folder contains the molecular dynamics simulation code to run the water simulation using SCFNN. It also contains the MD code to run the water simulation using the Behler-Parrinello model. It also cotains the code to do the finite-field constant D simulation using SCFNN.

Should you have any questions, please contact anggao@bupt.edu.cn
