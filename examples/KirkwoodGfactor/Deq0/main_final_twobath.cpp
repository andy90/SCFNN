//
//  main.cpp
//  MDML
//
//  Created by Ang Gao on 2021/8/2.
//

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <random>
#include <torch/script.h>
#include <algorithm>
#include "omp.h"

#define natoms 3000
#define noxygen 1000
#define nhydrogen 2000


using namespace std;
using namespace torch::indexing;

void initialize(vector<vector<float> > &pos, vector<vector<float> > &vel, vector<vector<float> > &acc, vector<float> &boxlength, float temperature, float massO, float massH);
void resume(vector<vector<float> > &pos, vector<vector<float> > &vel, vector<vector<float> > &acc, vector<float> &boxlength, float temperature, float massO, float massH);
void update(vector<vector<float> > &pos, vector<vector<float> > &vel, vector<vector<float> > &acc, vector<vector<float> > &force, float massO, float massH, float dt);
void updatepos(vector<vector<float> > &pos, vector<vector<float> > &vel, vector<vector<float> > &acc, vector<vector<float> > &force, float massO, float massH, float dt);
void updatevelocity_acc(vector<vector<float> > &pos, vector<vector<float> > &vel, vector<vector<float> > &acc, vector<vector<float> > &force, float massO, float massH, float dt);

void compute_force_GT(float rij[][natoms], float vecrij[][natoms][3], vector<vector<float> > &force, float massO, float massH, vector<vector<float> > &parameters2_OO, vector<vector<float> > &parameters2_OH, vector<vector<float> > &parameters2_HO, vector<vector<float> > &parameters2_HH, vector<vector<float> > &parameters4_OOO, vector<vector<float> > &parameters4_OOH, vector<vector<float> > &parameters4_OHH, vector<vector<float> > &parameters4_HHO, vector<vector<float> > &parameters4_HOO, float xOscaling[][3], float xHscaling[][3], torch::jit::script::Module &net);
void vel_scale(vector<vector<float> > &pos, float massO, float massH, float temperature, float tau1, float tau2, float dt);
void pbc(vector<vector<float> > &pos, vector<float> &boxlength);
void vel_pbc(vector<vector<float> > &pos, float massO, float massH);
void print_data(vector<vector<float> > &pos, int istep, vector<float> &box, ofstream &ffile);
void print_final(vector<vector<float> > &pos, vector<vector<float> > &vel, vector<vector<float> > &acc);
void printE(vector<vector<float> > & EO, vector<vector<float> > & EH, int istep, int iter, ofstream &ff);
void print_general(vector<vector<float> > & wxyz, int istep, int iter, ofstream &ff);

float fc(float r);
float dfc(float r);
float G2(float r12, float yeta, float rs);
float dG2(float r12, float yeta, float rs);
float G4(float r12, float r13, float r23, float cosalphaijk, float zeta, float yeta, float lam);
vector<vector<float> > dG4_ij_ik_jk(float r12, float r13, float r23, float cosalphaijk, vector<vector<float> > &parameters, int nx);
void read_parameters(vector<vector<float> >& parameters, string fp_name, int nx);
void get_G2features(float features[], vector<vector<float> > &parameters, int nx, float r[][natoms], int id_i, int id_j, float xscaling[][3]);
void get_dG2features(float features[], vector<vector<float> > &parameters, int nx, float r[][natoms], float vecr[][natoms][3], int id_i, int id_j, float xscaling[][3]);
void get_G4features(float features[],  vector<vector<float> > &parameters, int nx, float r[][natoms], int id_i, int id_j, int id_k, float xscaling[][3]);
void get_dG4features(float features[], vector<vector<float> > &parameters, int nx, float r[][natoms], float vecr[][natoms][3], int id_i, int id_j, int id_k, float xscaling[][3]);
void print_vector(vector<float> &a, string fname);
void read_scaling(float (&xOscaling)[30][3], float (&xHscaling)[27][3], float wannier_GT_feature_scale[][3], vector<float> & wannier_GT_target_scale);
void get_neighbourO(float r[][natoms], vector<vector<int> > &nO_list);
void get_neighbourH(float r[][natoms], vector<vector<int> > &nH_list);
vector<float> cross(vector<float> const &a, vector<float> const &b);
float norm(vector<float> const &a);
float dot(vector<float> const &a, vector<float> const &b);
vector<vector<float> > get_rotation(vector<float> const &x, vector<float> const &y);
vector<vector<float> > get_rotation2(vector<float> const &x, vector<float> const &yy);
void get_rotamers(vector<vector<vector<float> > > &vecrij_nn, vector<vector<vector<float> > > &rotamers);
void rotate(vector<vector<float> > &xyz, vector<vector<vector<float> > > &rotamers, float xyz_rotated[], int id_i, int id_j);
void shift_rotate(vector<vector<float> > &xyz, float xyz_shifted[][natoms][3], vector<vector<vector<float> > > &rotamers, float xyz_rotated[][natoms][3], int id_i);
void get_Ewald(vector<vector<float> > &pos, vector<vector<float> > &w_pos, vector<vector<float> > &EO, vector<vector<float> > &EH, vector<float> &boxlength, vector<float> Eext);
float G4new(float r12, float cosalphaijk, float zeta, float yeta, float lam);
float G4new_E(float r12, float cosalphaijk, float zeta, float yeta, float lam, float E);
float G2_E(float r12, float yeta, float rs, float E);
void get_dist(float rij[][natoms], float vecrij[][natoms][3], vector<vector<float> > &pos, vector<float> &boxlength);
void compute_wannier_GT(float rij[][natoms],  float pos_rotated[][natoms][3], vector<vector<vector<float> > > &wannier_mapped_rotated, vector<vector<float> > &parameters2_OO, vector<vector<float> > &parameters2_OH, vector<vector<float> > &parameters4_OO, vector<vector<float> > &parameters4_OH, float  wannier_GT_feature_scale[][3], vector<float> & wannier_GT_target_scale, torch::jit::script::Module &net);
void backrotate_shift_reshape(vector<vector<vector<float> > > &wannier_mapped_rotated, vector<vector<float> > & wxyz, vector<vector<float> > & pos, vector<vector<vector<float> > > &rotamers);
void backrotate_shift(vector<vector<vector<float> > > &wannier_mapped_rotated, vector<vector<vector<float> > > & wxyz, vector<vector<vector<float> > > &rotamers);
void compute_wannier_peturb(float rij[][natoms],  float pos_rotated[][natoms][3], vector<vector<vector<float> > > &wannier_peturb_mapped_rotated, float EO_rotated[], float EH_rotated[], vector<vector<float> > &parameters2_OO, vector<vector<float> > &parameters2_OH, vector<vector<float> > &parameters4_OO, vector<vector<float> > &parameters4_OH, torch::jit::script::Module &net);
void get_G2features_E(vector<vector<vector<float> > > &features, vector<vector<float> > &parameters, int nx, float r[][natoms], int id_i, int id_j, float E_rotated[]);
void get_G4new_features(vector<vector<vector<float> > > &features, vector<vector<float> > &parameters, int nx, float r[][natoms],  float vecrij[][natoms][3], int id_i, int id_j);
void get_G4new_features_E(vector<vector<vector<float> > > &features, vector<vector<float> > &parameters, int nx, float r[][natoms], float vecrij[][natoms][3], int id_i, int id_j, float E_rotated[]);
void compute_force_peturbO(float rij[][natoms], float pos_rotated[][natoms][3], vector<vector<float> >  &force, float EO_rotated[], float EH_rotated[], vector<vector<float> > &parameters2_OO, vector<vector<float> > &parameters2_OH, vector<vector<float> > &parameters4_OO, vector<vector<float> > &parameters4_OH, torch::jit::script::Module &net);
void compute_force_peturbH(float rij[][natoms], float pos_rotated[][natoms][3], vector<vector<float> >  &force, float EO_rotated[], float EH_rotated[], vector<vector<float> > &parameters2_HH, vector<vector<float> > &parameters2_HO, vector<vector<float> > &parameters4_HH, vector<vector<float> > &parameters4_HO, torch::jit::script::Module &net);
void backrotate(vector<vector<float> > &f_rotated, vector<vector<float> > & f_original, vector<vector<vector<float> > > &rotamers);
float get_diff_copy(vector<vector<float> > & data_last, vector<vector<float> > & data);
vector<float> get_dipole(vector<vector<vector<float> > > &wxyz_mapped, vector<vector<vector<float> > > &vecrij_nnO);
void get_time(int i){
    time_t time1 = time(NULL);
    printf("%d %s", i,ctime(&time1));
}


float pos_rotatedO[noxygen][natoms][3] {{{0}}};
float pos_rotatedH[nhydrogen][natoms][3] {{{0}}};
float rij[natoms][natoms];       // the distance matrix for the nucleus
float vecrij[natoms][natoms][3]; // the vector distance matrix
float EO_rotated[noxygen * noxygen *3] {0}; 
float EH_rotated[noxygen * nhydrogen * 3] {0};
            
float EO_rotated_H[nhydrogen * noxygen * 3] {0};
float EH_rotated_H[nhydrogen * nhydrogen * 3] {0};

float features_G2OO_fGT[8*noxygen];
float features_G2OH_fGT[8*noxygen];
float features_G2HO_fGT[8*nhydrogen];
float features_G2HH_fGT[8*nhydrogen];

float features_G4OOO_fGT[4*noxygen];
float features_G4OOH_fGT[4*noxygen];
float features_G4OHH_fGT[6*noxygen];
float features_G4HHO_fGT[7*nhydrogen];
float features_G4HOO_fGT[4*nhydrogen];
    
float features_dG2OO_fGT[8*noxygen*natoms*3];
float features_dG2OH_fGT[8*noxygen*natoms*3];
float features_dG2HO_fGT[8*nhydrogen*natoms*3];
float features_dG2HH_fGT[8*nhydrogen*natoms*3];
    
float features_dG4OOO_fGT[4*noxygen*natoms*3];
float features_dG4OOH_fGT[4*noxygen*natoms*3];
float features_dG4OHH_fGT[6*noxygen*natoms*3];
float features_dG4HHO_fGT[7*nhydrogen*natoms*3];
float features_dG4HOO_fGT[4*nhydrogen*natoms*3];


//float dSkO_real[noxygen][kdim][3];
//float dSkO_imag[noxygen][kdim][3];
//float dSkH_real[nhydrogen][kdim][3];
//float dSkH_imag[nhydrogen][kdim][3];



int main(int argc,char *argv[]) {
    omp_set_num_threads(20);

    vector<vector<float> > pos (natoms, vector<float>(3));
    vector<vector<float> > vel (natoms, vector<float>(3));
    vector<vector<float> > acc (natoms, vector<float>(3));
    vector<vector<float> > force (natoms, vector<float>(3));
    
    
    int nstep = atoi(argv[1]); // the number of timesteps
    float dt = atof(argv[2]); // this is in the atomic unit, corresponding to 1fs
    float temperature = atof(argv[3]); // the equil temperature
    float lambda1 = atof(argv[4]);
    float lambda2 = atof(argv[5]);
    float Ez = atof(argv[6])/51.4;
    int restart = atoi(argv[7]);
    
    float massO = 16 * 1822; // mass of oxygen in a.u.
    float massH = 1 * 1822; // mass of hydrogen in a.u.
    
    vector<float> boxlength(3);
    
    float tau1 = lambda1 * dt;
    float tau2 = lambda2 * dt;

    
    int istep;
    int step_rescale = 1;
    int step_print = 100;

    

    cout << "dt: " << dt << endl;
    cout << "temp scaling O lambda: " << lambda1 << endl;
    cout << "temp scaling H lambda: " << lambda2 << endl;
    cout << "nsteps: " << nstep << endl;
    cout << "Ez: " << Ez << endl;
    cout << "temp: " << temperature << endl;
    cout << "restart: " << restart << endl;
    
    ofstream fpos("xyz_hist.txt");
    ofstream fvel("vxyz_hist.txt");
    ofstream fforce("force_hist.txt");
    ofstream fEfield("E_hist.txt");
    ofstream fwxyz("wxyz_hist.txt");
    ofstream ffpeturb("force_peturb.txt");
    ofstream temp_Ecov_hist("temp_Econv_hist.txt");

    float xO_scaling[30][3];
    float xH_scaling[27][3];
    float wannier_GT_feature_scale[36][3];
    vector<float> wannier_GT_target_scale(12);
    read_scaling(xO_scaling, xH_scaling, wannier_GT_feature_scale, wannier_GT_target_scale);

    
    torch::jit::script::Module net_forceGT;
    net_forceGT = torch::jit::load("models/traced_trained_model.pt");

    torch::jit::script::Module net_wannierGT;
    net_wannierGT = torch::jit::load("models/traced_wannier_GT.pt");

    torch::jit::script::Module net_wannier_peturb;
    net_wannier_peturb = torch::jit::load("models/traced_wannier_peturb.pt");

    torch::jit::script::Module net_force_peturb_O;
    net_force_peturb_O= torch::jit::load("models/force_peturb_O.pt");
    
    torch::jit::script::Module net_force_peturb_H;
    net_force_peturb_H= torch::jit::load("models/force_peturb_H.pt");
    
    // parameters for the GT force
    vector<vector<float> > parameters2_OO(8, vector<float>(4));
    vector<vector<float> > parameters2_OH(8, vector<float>(4));
    vector<vector<float> > parameters2_HO(8, vector<float>(4));
    vector<vector<float> > parameters2_HH(8, vector<float>(4));

    vector<vector<float> > parameters4_OOO(4, vector<float>(4));
    vector<vector<float> > parameters4_OOH(4, vector<float>(4));
    vector<vector<float> > parameters4_OHH(6, vector<float>(4));
    vector<vector<float> > parameters4_HHO(7, vector<float>(4));
    vector<vector<float> > parameters4_HOO(4, vector<float>(4));
    
    read_parameters(parameters2_OO, "models/G2_parameters_OO.txt", 8);
    read_parameters(parameters2_OH, "models/G2_parameters_OH.txt", 8);
    read_parameters(parameters2_HO, "models/G2_parameters_HO.txt", 8);
    read_parameters(parameters2_HH, "models/G2_parameters_HH.txt", 8);

    read_parameters(parameters4_OOO, "models/G4_parameters_OOO.txt", 4);
    read_parameters(parameters4_OOH, "models/G4_parameters_OOH.txt", 4);
    read_parameters(parameters4_OHH, "models/G4_parameters_OHH.txt", 6);
    read_parameters(parameters4_HHO, "models/G4_parameters_HHO.txt", 7);
    read_parameters(parameters4_HOO, "models/G4_parameters_HOO.txt", 4);

    // parameters for the GT wannier center
    vector<vector<float> > parameters2_OO_w(6, vector<float>(4));
    vector<vector<float> > parameters2_OH_w(6, vector<float>(4));
    vector<vector<float> > parameters4_OO_w(4, vector<float>(4));
    vector<vector<float> > parameters4_OH_w(4, vector<float>(4));
    
    read_parameters(parameters2_OO_w, "models/wG2_parameters_OO.txt", 6);
    read_parameters(parameters2_OH_w, "models/wG2_parameters_OH.txt", 6);
    read_parameters(parameters4_OO_w, "models/wG4_parameters_OO.txt", 4);
    read_parameters(parameters4_OH_w, "models/wG4_parameters_OH.txt", 4);

    // parameters for the perturbation of the wannier center
    vector<vector<float> > parameters2_OO_dw(6, vector<float>(4));
    vector<vector<float> > parameters2_OH_dw(6, vector<float>(4));
    vector<vector<float> > parameters4_OO_dw(4, vector<float>(4));
    vector<vector<float> > parameters4_OH_dw(4, vector<float>(4));
    
    read_parameters(parameters2_OO_dw, "models/dwG2_parameters_OO.txt", 6);
    read_parameters(parameters2_OH_dw, "models/dwG2_parameters_OH.txt", 6);
    read_parameters(parameters4_OO_dw, "models/dwG4_parameters_OO.txt", 4);
    read_parameters(parameters4_OH_dw, "models/dwG4_parameters_OH.txt", 4);

    // parameters for the Oxygen force perturb
    vector<vector<float> > parameters2_OO_f(6, vector<float>(4));
    vector<vector<float> > parameters2_OH_f(6, vector<float>(4));
    vector<vector<float> > parameters4_OO_f(4, vector<float>(4));
    vector<vector<float> > parameters4_OH_f(4, vector<float>(4));
    
    read_parameters(parameters2_OO_f, "models/fG2_parameters_OO.txt", 6);
    read_parameters(parameters2_OH_f, "models/fG2_parameters_OH.txt", 6);
    read_parameters(parameters4_OO_f, "models/fG4_parameters_OO.txt", 4);
    read_parameters(parameters4_OH_f, "models/fG4_parameters_OH.txt", 4);

    // parameters for the Hydrogen force perturb
    vector<vector<float> > parameters2_HH_f(6, vector<float>(4));
    vector<vector<float> > parameters2_HO_f(6, vector<float>(4));
    vector<vector<float> > parameters4_HH_f(4, vector<float>(4));
    vector<vector<float> > parameters4_HO_f(4, vector<float>(4));
    
    read_parameters(parameters2_HH_f, "models/fG2_parameters_HH.txt", 6);
    read_parameters(parameters2_HO_f, "models/fG2_parameters_HO.txt", 6);
    read_parameters(parameters4_HH_f, "models/fG4_parameters_HH.txt", 4);
    read_parameters(parameters4_HO_f, "models/fG4_parameters_HO.txt", 4);
    
    for (istep = 0; istep < nstep; istep++){
        if (istep == 0){
            if (restart == 1){
                resume(pos, vel, acc, boxlength, temperature, massO, massH);
            }else{
                initialize(pos, vel, acc, boxlength, temperature, massO, massH);
            }
            
        }

        updatepos(pos, vel, acc, force, massO, massH, dt); // update the position of the verlocity verlet
        pbc(pos, boxlength);
        
        
        get_dist(rij, vecrij, pos, boxlength); // get the distance and vector dist
        // generate rotamers and rotate the position of the nucleus
        vector<vector<int> > nO_list(noxygen, vector<int>(2));  // for each oxygen, record the id of two closest H
        vector<vector<int> > nH_list(nhydrogen, vector<int>(2));  // for each hydrogen record the id of closest O and H

        get_neighbourO(rij, nO_list);  // get the neighbour list
        get_neighbourH(rij, nH_list);   
        
        vector<vector<vector<float> > > vecrij_nnO (noxygen, vector<vector<float> >(2, vector<float>(3)));  // the relative vector of the neighbor to the center atom
        vector<vector<vector<float> > > vecrij_nnH (nhydrogen, vector<vector<float> >(2, vector<float>(3)));

        for (int i = 0; i < noxygen; i++){
            for (int j = 0; j < 2; j++){
                for (int k = 0; k < 3; k++){
                    vecrij_nnO[i][j][k] = vecrij[i][nO_list[i][j]][k];
                }
            }
        }

        for (int i = 0; i < nhydrogen; i++){
            for (int j = 0; j < 2; j++){
                for (int k = 0; k < 3; k++){
                    vecrij_nnH[i][j][k] = vecrij[i + noxygen][nH_list[i][j]][k];
                }
            }
        }
        vector<vector<vector<float> > > RO (noxygen, vector<vector<float> >(3, vector<float>(3)));
        vector<vector<vector<float> > > RH (nhydrogen, vector<vector<float> >(3, vector<float>(3)));
        
        get_rotamers(vecrij_nnO, RO);  // get the rotamers
        get_rotamers(vecrij_nnH, RH);
        

        shift_rotate(pos, vecrij, RO, pos_rotatedO, 0);  // rotate the nucleus coordinate
        shift_rotate(pos, vecrij, RH, pos_rotatedH, 1);
        int nwannier = (4*noxygen);
        vector<vector<float> > wxyz(nwannier, vector<float>(3));
        vector<vector<vector<float> > > wxyz_mapped_rotated (noxygen, vector<vector<float> >(4, vector<float>(3)));
        vector<vector<vector<float> > > wxyz_mapped (noxygen, vector<vector<float> >(4, vector<float>(3)));
        compute_wannier_GT(rij, pos_rotatedO, wxyz_mapped_rotated, parameters2_OO_w, parameters2_OH_w, parameters4_OO_w, parameters4_OH_w, wannier_GT_feature_scale, wannier_GT_target_scale, net_wannierGT);

        backrotate_shift_reshape(wxyz_mapped_rotated, wxyz, pos, RO); // rotate back the wannier position for the GT system
        backrotate_shift(wxyz_mapped_rotated, wxyz_mapped, RO); // rotate back the wannier position for the GT system
        vector<float> total_dipole = get_dipole(wxyz_mapped, vecrij_nnO);

        vector<vector<float> > EO(noxygen, vector<float> (3));
        vector<vector<float> > EH(nhydrogen, vector<float> (3));
        
        vector<float> Eeffct(3);
        vector<float> Eext(3);
        for (int i = 0; i < 3; i++){        
            Eeffct[i] = -total_dipole[i] / (boxlength[0]*boxlength[1]*boxlength[2]) * 4 * M_PI;
            Eext[i] = Eeffct[i];
        }
        Eext[2] += Ez;
        cout << Eext[0] << " " << Eext[1] << " " << Eext[2] << endl;
        get_Ewald(pos, wxyz, EO, EH, boxlength, Eext);  // calculate the Ewald based on the nucleus and the wannier xyz
           
        rotate(EO, RO, EO_rotated, 0, 0);  // rotate the field around O
        rotate(EH, RO, EH_rotated, 0, 1);

        rotate(EO, RH, EO_rotated_H, 1, 0); // rotate the field around H
        rotate(EH, RH, EH_rotated_H, 1, 1);

        //printE(EO, EH, istep, 0, fEfield);
        //print_general(wxyz, istep, 0, fwxyz);

        vector<vector<float> > EO_last(noxygen, vector<float> (3));
        vector<vector<float> > EH_last(nhydrogen, vector<float> (3));
        vector<vector<float> > wxyz_last(4*noxygen, vector<float> (3));
        for (int i = 0; i < noxygen; i++){
            for (int j = 0; j< 3; j++){
                EO_last[i][j] = EO[i][j];
            }
        }
        for (int i = 0; i < nhydrogen; i++){
            for (int j = 0; j< 3; j++){
                EH_last[i][j] = EH[i][j];
            }
        }
        for (int i = 0; i < (4*noxygen); i++){
            for (int j = 0; j< 3; j++){
                wxyz_last[i][j] = wxyz[i][j];
            }
        }

        int iter=0;
        float EO_diff = 10;
        float EH_diff = 10;
        float wxyz_diff = 10;
        while (wxyz_diff > 0.0005){
            vector<vector<vector<float> > > wxyz_mapped_rotated_peturb (noxygen, vector<vector<float> >(4, vector<float>(3)));
            compute_wannier_peturb(rij, pos_rotatedO, wxyz_mapped_rotated_peturb, EO_rotated, EH_rotated, parameters2_OO_dw, parameters2_OH_dw, parameters4_OO_dw, parameters4_OH_dw, net_wannier_peturb);
            vector<vector<vector<float> > > wxyz_mapped_rotated_total (noxygen, vector<vector<float> >(4, vector<float>(3)));
            for (int i = 0; i < noxygen; i++){
                for (int j = 0; j < 4; j++){
                    for (int k = 0; k < 3; k++){
                        wxyz_mapped_rotated_total[i][j][k] = wxyz_mapped_rotated_peturb[i][j][k] + wxyz_mapped_rotated[i][j][k];
                    }
                }
            }
            //vector<vector<float> > wxyz(nwannier, vector<float>(3));
            backrotate_shift_reshape(wxyz_mapped_rotated_total, wxyz, pos, RO);
            backrotate_shift(wxyz_mapped_rotated_total, wxyz_mapped, RO);
            vector<float> total_dipole = get_dipole(wxyz_mapped, vecrij_nnO);
            vector<float> Eeffct(3);
            vector<float> Eext(3);
            for (int i = 0; i < 3; i++){        
                Eeffct[i] = -total_dipole[i] / (boxlength[0]*boxlength[1]*boxlength[2]) * 4 * M_PI;
                Eext[i] = Eeffct[i];
            }
            Eext[2] += Ez;
            cout << Eext[0] << " " << Eext[1] << " " << Eext[2] << endl;

            get_Ewald(pos, wxyz, EO, EH, boxlength, Eext); // get Ewald again

            EO_diff = get_diff_copy(EO_last, EO);
            EH_diff = get_diff_copy(EH_last, EH);
            wxyz_diff = get_diff_copy(wxyz_last, wxyz);
            temp_Ecov_hist << istep << "  " << iter << "  " << EO_diff << "  " << EH_diff << "  " << wxyz_diff << endl; 

            rotate(EO, RO, EO_rotated, 0, 0);  // rotate the field around O
            rotate(EH, RO, EH_rotated, 0, 1);

            rotate(EO, RH, EO_rotated_H, 1, 0); // rotate the field around H
            rotate(EH, RH, EH_rotated_H, 1, 1);

            iter++;
        }
        
        vector<vector<float> > fO_peturb (noxygen, vector<float>(3));
        vector<vector<float> > fH_peturb (nhydrogen, vector<float>(3));
        vector<vector<float> > fO_peturb_backrotate (noxygen, vector<float>(3));
        vector<vector<float> > fH_peturb_backrotate (nhydrogen, vector<float>(3));
        vector<vector<float> > fGT (natoms, vector<float>(3));

    
        compute_force_peturbO(rij, pos_rotatedO, fO_peturb, EO_rotated, EH_rotated, parameters2_OO_f, parameters2_OH_f, parameters4_OO_f, parameters4_OH_f, net_force_peturb_O); // for oxygen
  
        compute_force_peturbH(rij, pos_rotatedH, fH_peturb, EO_rotated_H, EH_rotated_H, parameters2_HH_f, parameters2_HO_f, parameters4_HH_f, parameters4_HO_f, net_force_peturb_H); // for hydrogen
        compute_force_GT(rij, vecrij, fGT, massO, massH, parameters2_OO, parameters2_OH, parameters2_HO, parameters2_HH, parameters4_OOO, parameters4_OOH, parameters4_OHH, parameters4_HHO, parameters4_HOO, xO_scaling, xH_scaling, net_forceGT);
        backrotate(fO_peturb, fO_peturb_backrotate, RO);
        backrotate(fH_peturb, fH_peturb_backrotate, RH);
        
        for (int i = 0; i < noxygen; i++){
            for (int j =0; j < 3; j++){
                force[i][j] = fGT[i][j] + fO_peturb_backrotate[i][j];
            }
        }
        
        for (int i = noxygen; i < natoms; i++){
            for (int j =0; j < 3; j++){
                force[i][j] = fGT[i][j] + fH_peturb_backrotate[i-noxygen][j];
            }
        }
        
        updatevelocity_acc(pos, vel, acc, force, massO, massH, dt); // update the velocity and accelaration 
        vel_pbc(vel, massO, massH);

        if (istep % step_rescale == 0){
            vel_scale(vel, massO, massH, temperature, tau1, tau2, dt);
        }
        
        float kinetic_energy_O = 0;
        float kinetic_energy_H = 0;
        float target_kinetic_energy = 1.5 * natoms * 3.16669e-6 * temperature;
    
        for (int i = 0; i < noxygen; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                kinetic_energy_O += 0.5 * massO * vel[i][j] * vel[i][j];
            }
        }
        for (int i = noxygen; i < natoms; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                kinetic_energy_H += 0.5 * massH * vel[i][j] * vel[i][j];
            }
        }
        temp_Ecov_hist << istep << "  " << kinetic_energy_O << "  " << kinetic_energy_H << "  " << target_kinetic_energy << endl;

        float kinetic_vc = 0;
        for (int i = 0; i < noxygen; i++)
        {
            int ind_H1 = 2*i + noxygen;
            int ind_H2 = 2*i + 1 + noxygen;

            float vc[3] {0};
            for (int j = 0; j < 3; j++){
                vc[j] +=  (massO * vel[i][j] + massH * vel[ind_H1][j] + + massH * vel[ind_H2][j])/(massO + 2*massH);
                kinetic_vc += 0.5 * (massO + 2*massH) * vc[j] * vc[j];
            } 
        }
        
        cout << istep << " " << kinetic_vc << endl;

        if (istep % step_print == 0){
            print_data(pos, istep, boxlength, fpos);
            //print_data(vel, istep, boxlength, fvel);
            //print_data(force, istep, boxlength, fforce);
            print_final(pos, vel, acc);
            //printE(EO, EH, istep, iter, fEfield);
            //printE(fO_peturb_backrotate, fH_peturb_backrotate, istep, iter, ffpeturb);
            print_general(wxyz, istep, iter, fwxyz);
        }
    }
    
    return 0;
}

void initialize(vector<vector<float> > &pos, vector<vector<float> > &vel, vector<vector<float> > &acc, vector<float> &boxlength, float temperature, float massO, float massH){
    // initialize the position, velocity and acceleration of the atoms
    ifstream fOxyz("Oxyz.txt"); // read in the xyz coordinates of the oxygen first
    ifstream fHxyz("Hxyz.txt"); // read in the xyz coordinates of the hydrogen next
    for (int i = 0; i < noxygen; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            fOxyz >> pos[i][j];
        }
    }
    for (int i = noxygen; i < natoms; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            fHxyz >> pos[i][j];
        }
    }
    fOxyz.close();
    fHxyz.close();
    
    ifstream fbox("box.txt");
    fbox >> boxlength[0]; // read in the boxlength of this configuration
    fbox >> boxlength[1];
    fbox >> boxlength[2];
    fbox.close();
    
    float kT = 3.16669e-6 * temperature;  //  kT in a.u.
    random_device rd;
    mt19937 gen(rd());
    //mt19937 gen(0);
    float vsigO = sqrt(kT / massO);
    float vsigH = sqrt(kT / massH);
    normal_distribution<float> dO(0, vsigO);
    normal_distribution<float> dH(0, vsigH);
    for (int i = 0; i < noxygen; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            vel[i][j] = dO(gen);
        }
    }
    
    for (int i = noxygen; i < natoms; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            vel[i][j] = dH(gen);
        }
    }
    
    
    for (int i = 0; i < natoms; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            acc[i][j] = 0.0;
        }
    }
}

void resume(vector<vector<float> > &pos, vector<vector<float> > &vel, vector<vector<float> > &acc, vector<float> &boxlength, float temperature, float massO, float massH){
    // initialize the position, velocity and acceleration of the atoms
    ifstream fOxyz("Oxyz.txt"); // read in the xyz coordinates of the oxygen first
    ifstream fHxyz("Hxyz.txt"); // read in the xyz coordinates of the hydrogen next
    ifstream fvxyz("vxyz.txt"); // read in velocity
    ifstream faxyz("axyz.txt"); // read in accelaration

    for (int i = 0; i < noxygen; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            fOxyz >> pos[i][j];
        }
    }
    for (int i = noxygen; i < natoms; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            fHxyz >> pos[i][j];
        }
    }
    for (int i = 0; i < natoms; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            fvxyz >> vel[i][j];
        }
    }
    fOxyz.close();
    fHxyz.close();
    fvxyz.close();
    
    ifstream fbox("box.txt");
    fbox >> boxlength[0]; // read in the boxlength of this configuration
    fbox >> boxlength[1];
    fbox >> boxlength[2];
    fbox.close();
    
    for (int i = 0; i < natoms; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            faxyz >> acc[i][j];
        }
    }
    faxyz.close();
}


void update(vector<vector<float> > &pos, vector<vector<float> > &vel, vector<vector<float> > &acc, vector<vector<float> > &force, float massO, float massH, float dt){
    //    UPDATE updates positions, velocities and accelerations.
    //    A velocity Verlet algorithm is used for the updating.
    //
    //    x(t+dt) = x(t) + v(t) * dt + 0.5 * a(t) * dt * dt
    //    v(t+dt) = v(t) + 0.5 * ( a(t) + a(t+dt) ) * dt
    //    a(t+dt) = f(t) / m
    for (int i = 0; i < noxygen; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            pos[i][j] += vel[i][j] * dt + 0.5 * acc[i][j] * dt * dt;
            vel[i][j] += 0.5 * dt * (acc[i][j] + force[i][j] / massO);
            acc[i][j] = force[i][j] / massO;
        }
    }
    for (int i = noxygen; i < natoms; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            pos[i][j] += vel[i][j] * dt + 0.5 * acc[i][j] * dt * dt;
            vel[i][j] += 0.5 * dt * (acc[i][j] + force[i][j] / massH);
            acc[i][j] = force[i][j] / massH;
        }
    }
}

void updatepos(vector<vector<float> > &pos, vector<vector<float> > &vel, vector<vector<float> > &acc, vector<vector<float> > &force, float massO, float massH, float dt){
    //    UPDATE updates positions, velocities and accelerations.
    //    A velocity Verlet algorithm is used for the updating.
    //
    //    x(t+dt) = x(t) + v(t) * dt + 0.5 * a(t) * dt * dt
    //    v(t+dt) = v(t) + 0.5 * ( a(t) + a(t+dt) ) * dt
    //    a(t+dt) = f(t) / m
    for (int i = 0; i < noxygen; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            pos[i][j] += vel[i][j] * dt + 0.5 * acc[i][j] * dt * dt;
        }
    }
    for (int i = noxygen; i < natoms; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            pos[i][j] += vel[i][j] * dt + 0.5 * acc[i][j] * dt * dt;
        }
    }
}

void updatevelocity_acc(vector<vector<float> > &pos, vector<vector<float> > &vel, vector<vector<float> > &acc, vector<vector<float> > &force, float massO, float massH, float dt){
    //    UPDATE updates positions, velocities and accelerations.
    //    A velocity Verlet algorithm is used for the updating.
    //
    //    x(t+dt) = x(t) + v(t) * dt + 0.5 * a(t) * dt * dt
    //    v(t+dt) = v(t) + 0.5 * ( a(t) + a(t+dt) ) * dt
    //    a(t+dt) = f(t) / m
    for (int i = 0; i < noxygen; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            vel[i][j] += 0.5 * dt * (acc[i][j] + force[i][j] / massO);
            acc[i][j] = force[i][j] / massO;
        }
    }
    for (int i = noxygen; i < natoms; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            vel[i][j] += 0.5 * dt * (acc[i][j] + force[i][j] / massH);
            acc[i][j] = force[i][j] / massH;
        }
    }
}

void vel_scale(vector<vector<float> > &vel, float massO, float massH, float temperature, float tau1, float tau2, float dt){
    // rescale the velocity to the given temperature
    float kinetic_energy_O = 0;
    float target_kinetic_energy_O = 1.5 * noxygen * 3.16669e-6 * temperature;
    
    for (int i = 0; i < noxygen; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            kinetic_energy_O += 0.5 * massO * vel[i][j] * vel[i][j];
        }
    }
    
    float scale_factor_O = sqrt(1 + dt/tau1*(target_kinetic_energy_O / kinetic_energy_O - 1));
    
    for (int i = 0; i < noxygen; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            vel[i][j] = vel[i][j] * scale_factor_O;
        }
    }

    float kinetic_energy_H = 0;
    float target_kinetic_energy_H = 1.5 * nhydrogen * 3.16669e-6 * temperature;
    
    for (int i = noxygen; i < natoms; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            kinetic_energy_H += 0.5 * massH * vel[i][j] * vel[i][j];
        }
    }
    
    float scale_factor_H = sqrt(1 + dt/tau2*(target_kinetic_energy_H / kinetic_energy_H - 1));
    
    for (int i = noxygen; i < natoms; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            vel[i][j] = vel[i][j] * scale_factor_H;
        }
    }

}


void pbc(vector<vector<float> > &pos, vector<float> &boxlength){
    // implement the periodic boundary condition. the box is centered around origin
    for (int i = 0; i < natoms; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            pos[i][j] = pos[i][j] - round( (pos[i][j] - 0.5*boxlength[j]) / boxlength[j]) * boxlength[j];
        }
    }
}

void vel_pbc(vector<vector<float> > &vel, float massO, float massH){
    // make sure that the total velocity along the x or y or z dierction is 0 
    float vx_total = 0;
    float vy_total = 0;
    float vz_total = 0;
    for (int i = 0; i < noxygen; i++){
        vx_total += massO * vel[i][0];
        vy_total += massO * vel[i][1];
        vz_total += massO * vel[i][2];
    }

    for (int i = noxygen; i < natoms; i++){
        vx_total += massH * vel[i][0];
        vy_total += massH * vel[i][1];
        vz_total += massH * vel[i][2];
    }

    vx_total /= (noxygen * massO + nhydrogen * massH);
    vy_total /= (noxygen * massO + nhydrogen * massH);
    vz_total /= (noxygen * massO + nhydrogen * massH);

    for (int i = 0; i < natoms; i++){
        vel[i][0] -= vx_total;
        vel[i][1] -= vy_total;
        vel[i][2] -= vz_total;
    }
}

void print_data(vector<vector<float> > &pos, int istep, vector<float> &box, ofstream &ffile){
    // print data to file
    ffile << istep << "  " << box[0] << "  " << 0 << " \n";
    ffile << istep << "  " << box[1] << "  " << 1 << " \n";
    ffile << istep << "  " << box[2] << "  " << 2 << " \n";
    for (int i = 0; i < natoms; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            ffile << pos[i][j] <<"  ";
        }
        ffile << endl;
    }
}

void print_final(vector<vector<float> > &pos, vector<vector<float> > &vel, vector<vector<float> > &acc){
    ofstream fO("Oxyz_final.txt");
    ofstream fH("Hxyz_final.txt");
    ofstream fv("vxyz_final.txt");
    ofstream fa("axyz_final.txt");

    for (int i = 0; i < noxygen; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            fO << pos[i][j] <<"  ";
        }
        fO << "\n";
    }

    for (int i = noxygen; i < natoms; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            fH << pos[i][j] <<"  ";
        }
        fH << "\n";
    }

    for (int i = 0; i < natoms; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            fv << vel[i][j] <<"  ";
        }
        fv << "\n";
    }

    for (int i = 0; i < natoms; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            fa << acc[i][j] <<"  ";
        }
        fa << "\n";
    }

    fO.close();
    fH.close();
    fv.close();
    fa.close();
}

void compute_force_GT(float rij[][natoms], float vecrij[][natoms][3], vector<vector<float> > &force, float massO, float massH, vector<vector<float> > &parameters2_OO, vector<vector<float> > &parameters2_OH, vector<vector<float> > &parameters2_HO, vector<vector<float> > &parameters2_HH, vector<vector<float> > &parameters4_OOO, vector<vector<float> > &parameters4_OOH, vector<vector<float> > &parameters4_OHH, vector<vector<float> > &parameters4_HHO, vector<vector<float> > &parameters4_HOO, float xOscaling[][3], float xHscaling[][3], torch::jit::script::Module &net){
        
    fill(begin(features_G2OO_fGT), end(features_G2OO_fGT), 0);
    fill(begin(features_G2OH_fGT), end(features_G2OH_fGT), 0);
    fill(begin(features_G2HO_fGT), end(features_G2HO_fGT), 0);
    fill(begin(features_G2HH_fGT), end(features_G2HH_fGT), 0);

    fill(begin(features_G4OOO_fGT), end(features_G4OOO_fGT), 0);
    fill(begin(features_G4OOH_fGT), end(features_G4OOH_fGT), 0);
    fill(begin(features_G4OHH_fGT), end(features_G4OHH_fGT), 0);
    fill(begin(features_G4HHO_fGT), end(features_G4HHO_fGT), 0);
    fill(begin(features_G4HOO_fGT), end(features_G4HOO_fGT), 0);

    fill(begin(features_dG2OO_fGT), end(features_dG2OO_fGT), 0);
    fill(begin(features_dG2OH_fGT), end(features_dG2OH_fGT), 0);
    fill(begin(features_dG2HO_fGT), end(features_dG2HO_fGT), 0);
    fill(begin(features_dG2HH_fGT), end(features_dG2HH_fGT), 0);

    fill(begin(features_dG4OOO_fGT), end(features_dG4OOO_fGT), 0);
    fill(begin(features_dG4OOH_fGT), end(features_dG4OOH_fGT), 0);
    fill(begin(features_dG4OHH_fGT), end(features_dG4OHH_fGT), 0);
    fill(begin(features_dG4HHO_fGT), end(features_dG4HHO_fGT), 0);
    fill(begin(features_dG4HOO_fGT), end(features_dG4HOO_fGT), 0);

    // get G2 features
    float (*pO)[3] = xOscaling;
    float (*pH)[3] = xHscaling;
    get_G2features(features_G2OO_fGT, parameters2_OO, 8, rij, 0, 0, pO);
    pO = pO + 8;
    get_G2features(features_G2OH_fGT, parameters2_OH, 8, rij, 0, 1, pO);
    pO = pO + 8;
    get_G2features(features_G2HH_fGT, parameters2_HH, 8, rij, 1, 1, pH);
    pH = pH + 8;
    get_G2features(features_G2HO_fGT, parameters2_HO, 8, rij, 1, 0, pH);
    pH = pH + 8;
    

    // get G4 features
    get_G4features(features_G4OOO_fGT, parameters4_OOO, 4, rij, 0, 0, 0, pO);
    pO = pO + 4;
    get_G4features(features_G4OOH_fGT, parameters4_OOH, 4, rij, 0, 0, 1, pO);
    pO = pO + 4;
    get_G4features(features_G4OHH_fGT, parameters4_OHH, 6, rij, 0, 1, 1, pO);
    get_G4features(features_G4HHO_fGT, parameters4_HHO, 7, rij, 1, 1, 0, pH);
    pH = pH + 7;
    get_G4features(features_G4HOO_fGT, parameters4_HOO, 4, rij, 1, 0, 0, pH);

    // get dG2 features
    float (*dpO)[3] = xOscaling;
    float (*dpH)[3] = xHscaling;
    get_dG2features(features_dG2OO_fGT, parameters2_OO, 8, rij, vecrij, 0, 0, dpO);
    dpO = dpO + 8;
    get_dG2features(features_dG2OH_fGT, parameters2_OH, 8, rij, vecrij, 0, 1, dpO);
    dpO = dpO + 8;
    get_dG2features(features_dG2HH_fGT, parameters2_HH, 8, rij, vecrij, 1, 1, dpH);
    dpH = dpH + 8;
    get_dG2features(features_dG2HO_fGT, parameters2_HO, 8, rij, vecrij, 1, 0, dpH);
    dpH = dpH + 8;

    // get dG4 features
    get_dG4features(features_dG4OOO_fGT, parameters4_OOO, 4, rij, vecrij, 0, 0, 0, dpO);
    dpO = dpO + 4;
    get_dG4features(features_dG4OOH_fGT, parameters4_OOH, 4, rij, vecrij, 0, 0, 1, dpO);
    dpO = dpO + 4;
    get_dG4features(features_dG4OHH_fGT, parameters4_OHH, 6, rij, vecrij, 0, 1, 1, dpO);
    get_dG4features(features_dG4HHO_fGT, parameters4_HHO, 7, rij, vecrij, 1, 1, 0, dpH);
    dpH = dpH + 7;
    get_dG4features(features_dG4HOO_fGT, parameters4_HOO, 4, rij, vecrij, 1, 0, 0, dpH);
    
    auto options = torch::TensorOptions().dtype(torch::kFloat);
    torch::Tensor features_G2OO_fGT_t = torch::from_blob(features_G2OO_fGT, {noxygen, 8}, options);
    torch::Tensor features_G2OH_fGT_t = torch::from_blob(features_G2OH_fGT, {noxygen, 8}, options);
    torch::Tensor features_G2HO_fGT_t = torch::from_blob(features_G2HO_fGT, {nhydrogen, 8}, options);
    torch::Tensor features_G2HH_fGT_t = torch::from_blob(features_G2HH_fGT, {nhydrogen, 8}, options);

    torch::Tensor features_G4OOO_fGT_t = torch::from_blob(features_G4OOO_fGT, {noxygen, 4}, options);
    torch::Tensor features_G4OOH_fGT_t = torch::from_blob(features_G4OOH_fGT, {noxygen, 4}, options);
    torch::Tensor features_G4OHH_fGT_t = torch::from_blob(features_G4OHH_fGT, {noxygen, 6}, options);
    torch::Tensor features_G4HHO_fGT_t = torch::from_blob(features_G4HHO_fGT, {nhydrogen, 7}, options);
    torch::Tensor features_G4HOO_fGT_t = torch::from_blob(features_G4HOO_fGT, {nhydrogen, 4}, options);
    
    torch::Tensor features_dG2OO_fGT_t = torch::from_blob(features_dG2OO_fGT, {noxygen, natoms, 3, 8}, options);
    torch::Tensor features_dG2OH_fGT_t = torch::from_blob(features_dG2OH_fGT, {noxygen, natoms, 3, 8}, options);
    torch::Tensor features_dG2HO_fGT_t = torch::from_blob(features_dG2HO_fGT, {nhydrogen, natoms, 3, 8}, options);
    torch::Tensor features_dG2HH_fGT_t = torch::from_blob(features_dG2HH_fGT, {nhydrogen, natoms, 3, 8}, options);
    
    torch::Tensor features_dG4OOO_fGT_t = torch::from_blob(features_dG4OOO_fGT, {noxygen, natoms, 3, 4}, options);
    torch::Tensor features_dG4OOH_fGT_t = torch::from_blob(features_dG4OOH_fGT, {noxygen, natoms, 3, 4}, options);
    torch::Tensor features_dG4OHH_fGT_t = torch::from_blob(features_dG4OHH_fGT, {noxygen, natoms, 3, 6}, options);
    torch::Tensor features_dG4HHO_fGT_t = torch::from_blob(features_dG4HHO_fGT, {nhydrogen, natoms, 3, 7}, options);
    torch::Tensor features_dG4HOO_fGT_t = torch::from_blob(features_dG4HOO_fGT, {nhydrogen, natoms, 3, 4}, options);

    torch::Tensor xO_tensor = torch::cat({features_G2OO_fGT_t, features_G2OH_fGT_t, features_G4OOO_fGT_t, features_G4OOH_fGT_t, features_G4OHH_fGT_t}, 1);
    torch::Tensor xH_tensor = torch::cat({features_G2HH_fGT_t, features_G2HO_fGT_t, features_G4HHO_fGT_t, features_G4HOO_fGT_t}, 1);
    torch::Tensor dxO_t = torch::cat({features_dG2OO_fGT_t, features_dG2OH_fGT_t, features_dG4OOO_fGT_t, features_dG4OOH_fGT_t, features_dG4OHH_fGT_t}, 3);
    torch::Tensor dxH_t = torch::cat({features_dG2HH_fGT_t, features_dG2HO_fGT_t, features_dG4HHO_fGT_t, features_dG4HOO_fGT_t}, 3);

    
    torch::Tensor dxOO_t = dxO_t.index({Slice(), Slice(None, noxygen), Slice(), Slice()});
    torch::Tensor dxHO_t = dxO_t.index({Slice(), Slice(noxygen, None), Slice(), Slice()});
    torch::Tensor dxHH_t = dxH_t.index({Slice(), Slice(noxygen, None), Slice(), Slice()});
    torch::Tensor dxOH_t = dxH_t.index({Slice(), Slice(None, noxygen), Slice(), Slice()});

    torch::Tensor xOO_d_tensor = dxOO_t.permute({1, 2, 0, 3});
    torch::Tensor xHO_d_tensor = dxHO_t.permute({1, 2, 0, 3});
    torch::Tensor xHH_d_tensor = dxHH_t.permute({1, 2, 0, 3});
    torch::Tensor xOH_d_tensor = dxOH_t.permute({1, 2, 0, 3});
   
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(xO_tensor);
    inputs.push_back(xH_tensor);
    inputs.push_back(xOO_d_tensor);
    inputs.push_back(xHO_d_tensor);
    inputs.push_back(xOH_d_tensor);
    inputs.push_back(xHH_d_tensor);
    
    auto outputs = net.forward(inputs).toTuple();
    torch::Tensor yO = outputs->elements()[0].toTensor();
    torch::Tensor yH = outputs->elements()[1].toTensor();
    
    for (int i = 0; i < noxygen; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            force[i][j] = yO.index({i, j}).item().toFloat() * 0.05;
        }
    }
    for (int i = noxygen; i < natoms; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            int i_offset = i - noxygen;
            force[i][j] = yH.index({i_offset, j}).item().toFloat() * 0.05;
        }
    }
}


float fc(float r)
{
    float rc = 12;
    float y = 0;
    if (r < rc)
    {
        y = pow(tanh(1 - r / rc), 3);
    }
    return y;
}

float dfc(float r)
{
    float rc = 12;
    float y = 0;
    if (r < rc)
    {
        y = -3 * pow(tanh(1 - r / rc), 2) / pow(cosh(1 - r / rc), 2) / rc;
    }
    return y;
}

float G2(float r12, float yeta, float rs)
{
    float y = exp(-yeta * (r12 - rs) * (r12 - rs)) * fc(r12);
    return y;
}

float dG2(float r12, float yeta, float rs) // this is only the radial part of dG2
{
    float y = -2 * yeta * (r12 - rs) * fc(r12) * exp(-yeta * pow((r12 - rs), 2)) + exp(-yeta * pow((r12 - rs), 2)) * dfc(r12);
    return y;
}

float G4(float r12, float r13, float r23, float cosalphaijk, float zeta, float yeta, float lam)
{
    float y = exp(-yeta * (r12 * r12 + r13 * r13 + r23 * r23)) * fc(r12) * fc(r13) * fc(r23) * pow((1 + lam * cosalphaijk), zeta);
    y = y * pow(2, 1 - zeta);
    return y;
}

vector<vector<float> > dG4_ij_ik_jk(float r12, float r13, float r23, float cosalphaijk, vector<vector<float> > &parameters, int nx)
{
    
    float fc12 = fc(r12);
    float fc13 = fc(r13);
    float fc23 = fc(r23);
    float dfc12 = dfc(r12);
    float dfc13 = dfc(r13);
    float dfc23 = dfc(r23);

    vector<vector<float> > ysp(nx, vector<float>(3));
    
    for (int ip = 0; ip < nx; ip++){
        float zeta = parameters[ip][3];
        float yeta = parameters[ip][1];
        float lam = parameters[ip][2];

        float commonvalue = pow(2, 1 - zeta) * exp(-yeta * (r12 * r12 + r13 * r13 + r23 * r23)) * pow((1 + lam * cosalphaijk), (zeta-1)) ;
    
        float y = 0;
        y += zeta * lam * ( 1.0/ r13 - (r12*r12 + r13*r13 - r23*r23) / (2*r12*r12*r13))  * fc12 * fc13 *fc23;
        y += - 2 * r12 * yeta  * (1 + lam * cosalphaijk) * fc12 * fc13 *fc23;
        y += dfc12 * (1 + lam * cosalphaijk) * fc13 *fc23;
        y = y * commonvalue;
        
        float y1 = 0;
        y1 += zeta * lam * ( 1.0/ r12 - (r12*r12 + r13*r13 - r23*r23) / (2*r13*r13*r12))  * fc12 * fc13 *fc23;
        y1 += - 2 * r13 * yeta  * (1 + lam * cosalphaijk) * fc12 * fc13 *fc23;
        y1 += dfc13 * (1 + lam * cosalphaijk) * fc12 *fc23;
        y1 = y1 * commonvalue;
        
        float y2 = 0;
        y2 += zeta * lam * (-r23/(r12 * r13)) * fc12 * fc13 *fc23;
        y2 += - 2 * r23 * yeta  * (1 + lam * cosalphaijk) * fc12 * fc13 *fc23;
        y2 += dfc23 * (1 + lam * cosalphaijk) * fc12 *fc13;
        y2 = y2 * commonvalue;

        ysp[ip][0] = y;
        ysp[ip][1] = y1;
        ysp[ip][2] = y2; 
    }
    
    return ysp;
}

void read_parameters(vector<vector<float> >& parameters, string fp_name, int nx)
{
    ifstream fp(fp_name);
    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            fp >> parameters[i][j];
        }
    }
}

void get_G2features(float features[], vector<vector<float> > &parameters, int nx, float r[][natoms], int id_i, int id_j, float xscaling[][3])
{
    float rc = 12;
    int rangei[2] = {0, noxygen};
    if (id_i == 1)
    {
        rangei[0] = noxygen;
        rangei[1] = natoms;
    }
    int rangej[2] = {0, noxygen};
    if (id_j == 1)
    {
        rangej[0] = noxygen;
        rangej[1] = natoms;
    }

    #pragma omp parallel for  
    for (int i = rangei[0]; i < rangei[1]; i++)
    {
        for (int j = rangej[0]; j < rangej[1]; j++)
        {
            if ((j != i) && (r[i][j] < rc))
            {

                for (int ip = 0; ip < nx; ip++)
                {
                    features[(i - id_i * noxygen) * nx + ip] += G2(r[i][j], parameters[ip][1], parameters[ip][0]);
                }
            }
        }

        for (int ip = 0; ip < nx; ip++)
        {
            features[(i - id_i * noxygen) * nx + ip] = (features[(i - id_i * noxygen) * nx + ip] - xscaling[ip][0])/(xscaling[ip][2] - xscaling[ip][1]) ;
        }
    }

    
}

void get_dG2features(float features[], vector<vector<float> > &parameters, int nx, float r[][natoms], float vecr[][natoms][3], int id_i, int id_j, float xscaling[][3])
{
    float rc = 12;
    int rangei[2] = {0, noxygen};
    if (id_i == 1)
    {
        rangei[0] = noxygen;
        rangei[1] = natoms;
    }
    int rangej[2] = {0, noxygen};
    if (id_j == 1)
    {
        rangej[0] = noxygen;
        rangej[1] = natoms;
    }

    #pragma omp parallel for      
    for (int i = rangei[0]; i < rangei[1]; i++)
    {
        for (int j = rangej[0]; j < rangej[1]; j++)
        {
            if ((j != i) && (r[i][j] < rc))
            {
                for (int ip = 0; ip < nx; ip++)
                {
                    float dG2ij = dG2(r[i][j], parameters[ip][1], parameters[ip][0]);
                    for (int ix = 0; ix < 3; ix++)
                    {
                        features[(i - id_i * noxygen)*natoms*nx*3 + j*nx*3 + ix*nx + ip] += dG2ij * vecr[i][j][ix] / r[i][j];
                        
                        features[(i - id_i * noxygen)*natoms*nx*3 + i*nx*3 + ix*nx + ip] -= dG2ij * vecr[i][j][ix] / r[i][j];
                        
                    }
                }
            }
        }

        for (int j = 0; j < natoms; j++)
        {
            for (int ip = 0; ip < nx; ip++)
            {
                for (int ix = 0; ix < 3; ix++)
                {
                    features[(i - id_i * noxygen)*natoms*nx*3 + j*nx*3 + ix*nx + ip] /= (xscaling[ip][2] - xscaling[ip][1]);
                                            
                }
            }
            
        }
    }

    
}

void get_G4features(float features[],  vector<vector<float> > &parameters, int nx, float r[][natoms], int id_i, int id_j, int id_k, float xscaling[][3])
{
    float rc = 12;
    int rangei[2] = {0, noxygen};
    if (id_i == 1)
    {
        rangei[0] = noxygen;
        rangei[1] = natoms;
    }
    int rangej[2] = {0, noxygen};
    if (id_j == 1)
    {
        rangej[0] = noxygen;
        rangej[1] = natoms;
    }
    int rangek[2] = {0, noxygen};
    if (id_k == 1)
    {
        rangek[0] = noxygen;
        rangek[1] = natoms;
    }

    #pragma omp parallel for  
    for (int i = rangei[0]; i < rangei[1]; i++)
    {
        for (int j = rangej[0]; j < rangej[1]; j++)
        {
            if ((j != i) && (r[i][j] < rc))
            {
                for (int k = rangek[0]; k < rangek[1]; k++)
                {
                    if ((k != j) && (k != i) && (r[k][i] < rc) && (r[k][j] < rc))
                    {
                        float cosijk = (r[i][j] * r[i][j] + r[i][k] * r[i][k] - r[j][k] * r[j][k]) / (2 * r[i][j] * r[i][k]);
                        for (int ip = 0; ip < nx; ip++)
                        {
                            features[(i - id_i * noxygen) * nx + ip] += G4(r[i][j], r[i][k], r[j][k], cosijk, parameters[ip][3], parameters[ip][1], parameters[ip][2]);
                        }
                    }
                }
            }
        }

        for (int ip = 0; ip < nx; ip++)
        {
            features[(i - id_i * noxygen) * nx + ip] = (features[(i - id_i * noxygen) * nx + ip] - xscaling[ip][0])/(xscaling[ip][2] - xscaling[ip][1]) ;
        }
    }

    
}

void get_dG4features(float features[], vector<vector<float> > &parameters, int nx, float r[][natoms], float vecr[][natoms][3], int id_i, int id_j, int id_k, float xscaling[][3])
{
    float rc = 12;
    int rangei[2] = {0, noxygen};
    if (id_i == 1)
    {
        rangei[0] = noxygen;
        rangei[1] = natoms;
    }
    int rangej[2] = {0, noxygen};
    if (id_j == 1)
    {
        rangej[0] = noxygen;
        rangej[1] = natoms;
    }
    int rangek[2] = {0, noxygen};
    if (id_k == 1)
    {
        rangek[0] = noxygen;
        rangek[1] = natoms;
    }

    #pragma omp parallel for  
    for (int i = rangei[0]; i < rangei[1]; i++)
    {
        for (int j = rangej[0]; j < rangej[1]; j++)
        {
            if ((j != i) && (r[i][j] < rc))
            {
                for (int k = rangek[0]; k < rangek[1]; k++)
                {
                    if ((k != j) && (k != i) && (r[k][i] < rc) && (r[k][j] < rc))
                    {
                        float cosijk = (r[i][j] * r[i][j] + r[i][k] * r[i][k] - r[j][k] * r[j][k]) / (2 * r[i][j] * r[i][k]);
                        vector<vector<float> > Gs = dG4_ij_ik_jk(r[i][j], r[i][k], r[j][k], cosijk, parameters, nx);
                        
                        for (int ip = 0; ip < nx; ip++)
                        {
                            
                            float dG4_ij_ij = Gs[ip][0];
                            float dG4_ij_ik = Gs[ip][1];
                            float dG4_jk_jk = Gs[ip][2];
                            for (int ix = 0; ix < 3; ix++)
                            {
                                features[(i - id_i * noxygen)*natoms*nx*3 + j*nx*3 + ix*nx + ip] += dG4_ij_ij * vecr[i][j][ix] / r[i][j];
                                features[(i - id_i * noxygen)*natoms*nx*3 + j*nx*3 + ix*nx + ip] += dG4_jk_jk * vecr[k][j][ix] / r[j][k];
                                
                                features[(i - id_i * noxygen)*natoms*nx*3 + k*nx*3 + ix*nx + ip] += dG4_ij_ik * vecr[i][k][ix] / r[i][k];
                                features[(i - id_i * noxygen)*natoms*nx*3 + k*nx*3 + ix*nx + ip] += dG4_jk_jk * vecr[j][k][ix] / r[k][j];
                                
                                features[(i - id_i * noxygen)*natoms*nx*3 + i*nx*3 + ix*nx + ip] -= dG4_ij_ij * vecr[i][j][ix] / r[i][j];
                                features[(i - id_i * noxygen)*natoms*nx*3 + i*nx*3 + ix*nx + ip] -= dG4_ij_ik * vecr[i][k][ix] / r[i][k];;
                            }
                        }
                    }
                }
            }
        }

        for (int j = 0; j < natoms; j++)
        {
            for (int ip = 0; ip < nx; ip++)
            {
                for (int ix = 0; ix < 3; ix++)
                {
                    features[(i - id_i * noxygen)*natoms*nx*3 + j*nx*3 + ix*nx + ip] /= (xscaling[ip][2] - xscaling[ip][1]);
                                            
                }
            }
            
        }
    }

    

}

void print_vector(vector<float> &a, string fname){
    ofstream ffile(fname);
    for (int i = 0; i < a.size(); i++){
        ffile << a[i] <<" ";
    }
    ffile.close();
}

void read_scaling(float (&xOscaling)[30][3], float (&xHscaling)[27][3], float wannier_GT_feature_scale[][3], vector<float> & wannier_GT_target_scale){
    ifstream fxO("models/xO_scalefactor.txt");
    for (int i = 0; i < 30; i++){
        for (int j = 0; j < 3; j++){
            fxO >> xOscaling[i][j];
        }
    }
    
    ifstream fxH("models/xH_scalefactor.txt");
    for (int i = 0; i < 27; i++){
        for (int j = 0; j < 3; j++){
            fxH >> xHscaling[i][j];
        }
    }

    ifstream fw("models/wannier_GT_feature_scale.txt");
    for (int i = 0; i < 36; i++){
        for (int j = 0; j < 3; j++){
            fw >> wannier_GT_feature_scale[i][j];
        }
    }

    ifstream tw("models/wannier_GT_target_scale.txt");
    for (int i = 0; i < wannier_GT_target_scale.size(); i++){
        tw >> wannier_GT_target_scale[i];
    }

}

// find out the two nearest Hydrogens relative to the Oxygen. return their id
void get_neighbourO(float r[][natoms], vector<vector<int> > &nO_list){
    float rO_list[noxygen][2];
    for (int i = 0; i < noxygen; i++){
        rO_list[i][0] = 1000;
        rO_list[i][1] = 10000;
        nO_list[i][0] = 1000;
        nO_list[i][1] = 10000;
    }

    #pragma omp parallel for
    for (int i = 0; i < noxygen; i++){
        for (int j = noxygen; j < natoms; j++){
            if (r[i][j] < rO_list[i][0]){
                rO_list[i][1] = rO_list[i][0];
                nO_list[i][1] = nO_list[i][0];

                rO_list[i][0] = r[i][j];
                nO_list[i][0] = j;
            }else if ((r[i][j] >= rO_list[i][0]) && (r[i][j] < rO_list[i][1])){
                rO_list[i][1] = r[i][j];
                nO_list[i][1] = j;
            }
        }
    }

}

// find out the two nearest Hydrogens relative to the Oxygen. return their id
void get_neighbourH(float r[][natoms], vector<vector<int> > &nH_list){
    float rH_list[nhydrogen][2];
    for (int i = 0; i < nhydrogen; i++){
        rH_list[i][0] = 1000;
        rH_list[i][1] = 10000;
        nH_list[i][0] = 1000;
        nH_list[i][1] = 10000;
    }

    #pragma omp parallel for
    for (int i = noxygen; i < natoms; i++){
        for (int j = 0; j < noxygen; j++){
            if (r[i][j] < rH_list[i - noxygen][0]){
                rH_list[i - noxygen][0] = r[i][j];
                nH_list[i - noxygen][0] = j;
            }
        }

        for (int j = noxygen; j < natoms; j++){
            if ((r[i][j] < rH_list[i - noxygen][1]) && (r[i][j] > 0.000000001)){
                rH_list[i - noxygen][1] = r[i][j];
                nH_list[i - noxygen][1] = j;
            }
        }
    }
}

vector<float> cross(vector<float> const &a, vector<float> const &b)
{
  vector<float> r (3);  
  r[0] = a[1]*b[2]-a[2]*b[1];
  r[1] = a[2]*b[0]-a[0]*b[2];
  r[2] = a[0]*b[1]-a[1]*b[0];
  return r;
}

float dot(vector<float> const &a, vector<float> const &b)
{
  float sum =  a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  return sum;
}

float norm(vector<float> const &a){
    float sum = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
    return sum;
}

vector<vector<float> > get_rotation(vector<float> const &x, vector<float> const &yy){
    vector<float> nx {x[0] / norm(x), x[1]/norm(x), x[2]/norm(x)};
    vector<float> y {yy[0] / norm(yy), yy[1]/norm(yy), yy[2]/norm(yy)};

    vector<float> c_nx_y = cross(nx, y);
    vector<float> new_ax (3);
    for (int i = 0; i < 3; i++){
        new_ax[i] = c_nx_y[i] / norm(c_nx_y);
    }
    if (norm(c_nx_y) == 0){
        vector<vector<float> > Ro(3, vector<float>(3));
        float sign = dot(nx, y);
        for (int i = 0; i < 3; i++){
            for (int j = 0; j < 3; j++){
                if (i == j){
                    Ro[i][j] += sign;
                }
            }
        }
        return Ro;
    }
    float d_nx_y = dot(nx, y);
    float phi = acos(d_nx_y);

    float W[3][3];
    W[0][0] = 0;
    W[1][0] = new_ax[2];
    W[2][0] = -new_ax[1];
    W[0][1] = -new_ax[2];
    W[1][1] = 0;
    W[2][1] = new_ax[0];
    W[0][2] = new_ax[1];
    W[1][2] = -new_ax[0];
    W[2][2] = 0;

    vector<vector<float> > Ro(3, vector<float>(3));
    for (int i = 0; i < 3; i++){
        for (int j = 0; j < 3; j++){
            if (i == j){
                Ro[i][j] += 1;
            }

            Ro[i][j] += sin(phi) * W[i][j];

            for (int k = 0; k < 3; k++){
                Ro[i][j] += 2 * pow(sin(phi / 2.0), 2) * W[i][k] * W[k][j];
            }
        }
    }
    return Ro;
}


vector<vector<float> > get_rotation2(vector<float> const &x, vector<float> const &yy){
    vector<float> nx {x[0] / norm(x), x[1]/norm(x), x[2]/norm(x)};
    vector<float> y {yy[0] / norm(yy), yy[1]/norm(yy), yy[2]/norm(yy)};

    vector<float> new_ax (3);
    new_ax[2] = 1;

    vector<float> c_nx_y = cross(nx, y);
    
    if (norm(c_nx_y) != 0){
        for (int i = 0; i < 3; i++){
            new_ax[i] = c_nx_y[i] / norm(c_nx_y);
        }
    }
    float d_nx_y = dot(nx, y);
    float phi = acos(d_nx_y);

    float W[3][3];
    W[0][0] = 0;
    W[1][0] = new_ax[2];
    W[2][0] = -new_ax[1];
    W[0][1] = -new_ax[2];
    W[1][1] = 0;
    W[2][1] = new_ax[0];
    W[0][2] = new_ax[1];
    W[1][2] = -new_ax[0];
    W[2][2] = 0;

    vector<vector<float> > Ro(3, vector<float>(3));
    for (int i = 0; i < 3; i++){
        for (int j = 0; j < 3; j++){
            if (i == j){
                Ro[i][j] += 1;
            }

            Ro[i][j] += sin(phi) * W[i][j];

            for (int k = 0; k < 3; k++){
                Ro[i][j] += 2 * pow(sin(phi / 2.0), 2) * W[i][k] * W[k][j];
            }
        }
    }
    return Ro;
}

void get_rotamers(vector<vector<vector<float> > > &vecrij_nn, vector<vector<vector<float> > > &rotamers){
    
    for (int i = 0; i < vecrij_nn.size(); i++){
        vector<float> vec_nn1 (3);
        vector<float> vec_nn2 (3);
        for (int j = 0; j < 3; j++){
            vec_nn1[j] = vecrij_nn[i][0][j];
            vec_nn2[j] = vecrij_nn[i][1][j];
        }

        float vec1_len = norm(vec_nn1);
        

        vector<float> zax(3), xax(3);
        vector<float> xax_unscale = cross(vec_nn2, vec_nn1);
        float xu_len = norm(xax_unscale);
        for (int j = 0; j < 3; j++){
            zax[j] = vec_nn1[j] / vec1_len;
            xax[j] = xax_unscale[j] / xu_len;
        }
        vector<float> yax = cross(zax, xax);

        for (int j = 0; j < 3; j++){
            rotamers[i][0][j] = xax[j];
            rotamers[i][1][j] = yax[j];
            rotamers[i][2][j] = zax[j];
        }
    }

}

void rotate(vector<vector<float> > &xyz, vector<vector<vector<float> > > &rotamers, float xyz_rotated[], int id_i, int id_j){
    int nx = noxygen;
    if (id_i == 1){
        nx = nhydrogen;
    }
    int ny = noxygen;
    if (id_j == 1){
        ny = nhydrogen;
    }

    for (int i=0; i<nx; i++){
        for (int j=0; j<ny; j++){
            for (int k=0; k<3; k++){
                xyz_rotated[i * 3 * ny + j*3 + k] = 0;
                for (int kk=0; kk<3; kk++){
                    xyz_rotated[i * 3 * ny + j*3 + k] += rotamers[i][k][kk] * xyz[j][kk];
                }
            }
            
        }
    }
}

void shift_rotate(vector<vector<float> > &xyz, float xyz_shifted[][natoms][3], vector<vector<vector<float> > > &rotamers, float xyz_rotated[][natoms][3], int id_i){

    
    int nx = noxygen;
    if (id_i == 1){
        nx = nhydrogen;
    }
    
    #pragma omp parallel for
    for (int i=0; i<nx; i++){
        int ii = i;
        if (id_i == 1){
            ii += noxygen;
        }
        for (int j=0; j<natoms; j++){
            for (int k=0; k<3; k++){
                xyz_rotated[i][j][k] = 0;
                for (int kk=0; kk<3; kk++){
                    xyz_rotated[i][j][k] += rotamers[i][k][kk] * xyz_shifted[ii][j][kk];
                }
            }
            
        }
    }
}

void get_Ewald(vector<vector<float> > &pos, vector<vector<float> > &w_pos, vector<vector<float> > &EO, vector<vector<float> > &EH, vector<float> &boxlength, vector<float> Eext){
    float qO = 6;
    float qH = 1;
    float qw = -2;
    float sigma = 8;
    float sigmaE = 8 / sqrt(2);
    float k0x = 2 * M_PI / boxlength[0];
    float k0y = 2 * M_PI / boxlength[1];
    float k0z = 2 * M_PI / boxlength[2];

    int nkmaxx = 4;
    int nkmaxy = 4;
    int nkmaxz = 4;
    int kdim = (2*nkmaxx - 1) * (2*nkmaxy - 1) * (2*nkmaxz - 1) - 1;
    
    vector<vector<float> > kxyz (kdim, vector<float>(3));
    int ik = 0;
    for (int i = (- nkmaxx + 1); i < nkmaxx; i++){
        for (int j = (- nkmaxy + 1); j < nkmaxy; j++){
            for (int k = (- nkmaxz + 1); k< nkmaxz; k++){
                if ((i != 0) | (j != 0) | (k != 0)){
                    kxyz[ik][0] = i * k0x;
                    kxyz[ik][1] = j * k0y;
                    kxyz[ik][2] = k * k0z;
                    ik++;
                }
            }
        }
    }

    vector<float> Sk_real(kdim);
    vector<float> Sk_imag(kdim);

    
    //memset(dSkO_real, 0, sizeof(dSkO_real));
    //memset(dSkO_imag, 0, sizeof(dSkO_imag));
    //memset(dSkH_real, 0, sizeof(dSkH_real));
    //memset(dSkH_imag, 0, sizeof(dSkH_imag));
    

    #pragma omp parallel for
    for (int i = 0; i < kdim; i++){
        for (int j = 0; j < noxygen; j++){
            float k_r_prod = 0;
            for (int k=0; k<3; k++){
                k_r_prod += kxyz[i][k] * pos[j][k]; 
            }
            Sk_real[i] += qO * cos(k_r_prod);
            Sk_imag[i] += qO * sin(k_r_prod);
        }

        for (int j = noxygen; j < natoms; j++){
            float k_r_prod = 0;
            for (int k=0; k<3; k++){
                k_r_prod += kxyz[i][k] * pos[j][k]; 
            }
            Sk_real[i] += qH * cos(k_r_prod);
            Sk_imag[i] += qH * sin(k_r_prod);
        }

        for (int j = 0; j < w_pos.size(); j++){
            float k_r_prod = 0;
            for (int k=0; k<3; k++){
                k_r_prod += kxyz[i][k] * w_pos[j][k]; 
            }
            Sk_real[i] += qw * cos(k_r_prod);
            Sk_imag[i] += qw * sin(k_r_prod);
        }
    }

    double coeff = 2 * M_PI / (boxlength[0] * boxlength[1] * boxlength[2]);
    
    vector<float> knorm (kdim);
    for(int i = 0; i < kdim; i++){
        vector<float> ki {kxyz[i][0], kxyz[i][1], kxyz[i][2]};
        knorm[i] = norm(ki);
    }

    for (int i =0;  i<noxygen; i++){      
        for (int k = 0; k < 3; k++){
            EO[i][k] = Eext[k];
        }
    }

    for (int i =0;  i<nhydrogen; i++){      
        for (int k = 0; k < 3; k++){
            EH[i][k] = Eext[k];
        }
    }

    #pragma omp parallel for
    for (int i =0;  i<noxygen; i++){     
        for (int j = 0; j < kdim; j++)
        {   
            float k_r_prod = 0;
            for (int k=0; k<3; k++){
                k_r_prod += kxyz[j][k] * pos[i][k]; 
            }
            
            for (int k = 0; k < 3; k++){
                float dSkO_real = - kxyz[j][k] * sin(k_r_prod);
                float dSkO_imag = kxyz[j][k] * cos(k_r_prod);
                EO[i][k] += -coeff * exp(-pow(sigmaE * knorm[j], 2)/2)/ pow(knorm[j], 2) * 2 * (Sk_real[j] * dSkO_real + Sk_imag[j] * dSkO_imag);
            }
            
        } 
    }
    #pragma omp parallel for
    for (int i =0;  i<nhydrogen; i++){
        for (int j = 0; j < kdim; j++)
        {   
            float k_r_prod = 0;
            for (int k=0; k<3; k++){
                k_r_prod += kxyz[j][k] * pos[i + noxygen][k]; 
            }

            for (int k = 0; k < 3; k++){
                float dSkH_real = - kxyz[j][k] * sin(k_r_prod);
                float dSkH_imag = kxyz[j][k] * cos(k_r_prod);
                EH[i][k] += -coeff * exp(-pow(sigmaE * knorm[j], 2)/2)/ pow(knorm[j], 2) * 2 * (Sk_real[j] * dSkH_real + Sk_imag[j] * dSkH_imag);
            }
            
        } 
    }
}

float G4new(float r12, float cosalphaijk, float zeta, float yeta, float lam)
{
    float y = exp(-yeta * (r12 * r12)) * fc(r12) * pow((1 + lam * cosalphaijk), zeta);
    y = y * pow(2, 1 - zeta);
    return y;
}

float G4new_E(float r12, float cosalphaijk, float zeta, float yeta, float lam, float E)
{
    float y = E * exp(-yeta * (r12 * r12)) * fc(r12) * pow((1 + lam * cosalphaijk), zeta);
    y = y * pow(2, 1 - zeta);
    return y;
}

float G2_E(float r12, float yeta, float rs, float E)
{
    float y = E * exp(-yeta * (r12 - rs) * (r12 - rs)) * fc(r12);
    return y;
}

// to be debugged. 8/3/2021
void get_G4new_features(vector<vector<vector<float> > > &features, vector<vector<float> > &parameters, int nx, float r[][natoms], float vecrij[][natoms][3], int id_i, int id_j)
{
    float rc = 12;
    int rangei[2] = {0, noxygen};
    if (id_i == 1)
    {
        rangei[0] = noxygen;
        rangei[1] = natoms;
    }
    int rangej[2] = {0, noxygen};
    if (id_j == 1)
    {
        rangej[0] = noxygen;
        rangej[1] = natoms;
    }

    for (int i = rangei[0]; i < rangei[1]; i++)
    {
        for (int j = rangej[0]; j < rangej[1]; j++)
        {
            if ((j != i) && (r[i][j] < rc)){
                for (int k = 0; k < 3; k++){
                
                float cosalpha = vecrij[i - id_i * noxygen][j][k]/r[i][j]; 
                    for (int ip = 0; ip < nx; ip++)
                    {
                    features[ip][i - id_i * noxygen][k] += G4new(r[i][j], cosalpha, parameters[ip][3], parameters[ip][1], parameters[ip][2]);
                    }
                }
            }
        }
    }
}

void get_G4new_features_E(vector<vector<vector<float> > > &features, vector<vector<float> > &parameters, int nx, float r[][natoms], float vecrij[][natoms][3], int id_i, int id_j, float E_rotated[])
{
    float rc = 12;
    int rangei[2] = {0, noxygen};
    int nEx = noxygen;
    if (id_i == 1)
    {
        rangei[0] = noxygen;
        rangei[1] = natoms;
        nEx = nhydrogen;
    }
    int rangej[2] = {0, noxygen};
    int nEy = noxygen;
    if (id_j == 1)
    {
        rangej[0] = noxygen;
        rangej[1] = natoms;
        nEy = nhydrogen;
    }

    for (int i = rangei[0]; i < rangei[1]; i++)
    {
        for (int j = rangej[0]; j < rangej[1]; j++)
        {
            if ((j != i) && (r[i][j] < rc)){
                for (int k = 0; k < 3; k++){
                    for (int l = 0; l < 3; l++){
                        float cosalpha = vecrij[i - id_i * noxygen][j][k]/r[i][j]; 
                        for (int ip = 0; ip < nx; ip++)
                        {
                        features[ip][i - id_i * noxygen][3 * k + l] += G4new_E(r[i][j], cosalpha, parameters[ip][3], parameters[ip][1], parameters[ip][2], E_rotated[(i - id_i * noxygen)*nEy*3 + (j - id_j * noxygen)*3 +l]);
                        }
                    }
                }
            }
        }
    }
}

void get_G2features_E(vector<vector<vector<float> > > &features, vector<vector<float> > &parameters, int nx, float r[][natoms], int id_i, int id_j, float E_rotated[])
{
    float rc = 12;
    int rangei[2] = {0, noxygen};
    int nEx = noxygen;
    if (id_i == 1)
    {
        rangei[0] = noxygen;
        rangei[1] = natoms;
        nEx = nhydrogen;
    }
    int rangej[2] = {0, noxygen};
    int nEy = noxygen;
    if (id_j == 1)
    {
        rangej[0] = noxygen;
        rangej[1] = natoms;
        nEy = nhydrogen;
    }

    for (int i = rangei[0]; i < rangei[1]; i++)
    {
        for (int j = rangej[0]; j < rangej[1]; j++)
        {
            if ((j != i) && (r[i][j] < rc)){
                for (int k = 0; k < 3; k++){
                    for (int ip = 0; ip < nx; ip++)
                    {
                        features[ip][i - id_i * noxygen][k] += G2_E(r[i][j], parameters[ip][1], parameters[ip][0], E_rotated[(i - id_i * noxygen)*nEy*3 + (j - id_j * noxygen)*3 +k]);
                    }
                }
            }
        }
    }
}

void get_dist(float rij[][natoms], float vecrij[][natoms][3], vector<vector<float> > &pos, vector<float> &boxlength){
    // get rij and vecrij
    #pragma omp parallel for
    for (int i = 0; i < natoms; i++)
    {
        for (int j = 0; j < natoms; j++)
        {
            float disx =  pos[j][0] -  pos[i][0] - round(( pos[j][0] -  pos[i][0]) / boxlength[0]) * boxlength[0];
            float disy =  pos[j][1] -  pos[i][1] - round(( pos[j][1] -  pos[i][1]) / boxlength[1]) * boxlength[1];
            float disz =  pos[j][2] -  pos[i][2] - round(( pos[j][2] -  pos[i][2]) / boxlength[2]) * boxlength[2];

            float dis = sqrt(disx * disx + disy * disy + disz * disz);
            rij[i][j] = dis;
            vecrij[i][j][0] = disx;
            vecrij[i][j][1] = disy;
            vecrij[i][j][2] = disz;
        }
    }

}

void compute_wannier_GT(float rij[][natoms], float pos_rotated[][natoms][3], vector<vector<vector<float> > > &wannier_mapped_rotated, vector<vector<float> > &parameters2_OO, vector<vector<float> > &parameters2_OH, vector<vector<float> > &parameters4_OO, vector<vector<float> > &parameters4_OH, float wannier_GT_feature_scale[][3], vector<float> & wannier_GT_target_scale, torch::jit::script::Module &net){
    vector<float> features_G2OO_wGT (6 *noxygen);
    vector<float> features_G2OH_wGT (6 *noxygen);
    vector<vector<vector<float> > > features_G4OO(4, vector<vector<float> >(noxygen, vector<float> (3)));
    vector<vector<vector<float> > > features_G4OH(4, vector<vector<float> >(noxygen, vector<float> (3)));
 
    // get G2 features
    float (*pO)[3] = wannier_GT_feature_scale;
    get_G2features(&features_G2OO_wGT[0], parameters2_OO, 6, rij, 0, 0, pO);
    pO = pO + 6;
    get_G2features(&features_G2OH_wGT[0], parameters2_OH, 6, rij, 0, 1, pO);

    // get G4 features
    get_G4new_features(features_G4OO, parameters4_OO, 4, rij, pos_rotated, 0, 0);
    get_G4new_features(features_G4OH, parameters4_OO, 4, rij, pos_rotated, 0, 1);

    // assemble the features into desired shape
    int nfeatures_O = 6 + 6 + 4*3 + 4*3;
    vector<float> xO(noxygen * nfeatures_O);
    
    int counter = 0;
    for (int i = 0; i < noxygen; i++){
        for (int ip = 0; ip < 6; ip++){
            xO[counter] = features_G2OO_wGT[i*6 + ip];
            counter++;
        }
        for (int ip = 0; ip < 6; ip++){
            xO[counter] = features_G2OH_wGT[i*6 + ip];
            counter++;
        }
        for (int j = 0; j < 3; j++){
            for (int ip = 0; ip < 4; ip++){
                xO[counter] = features_G4OO[ip][i][j];
                counter++;
            }
        }
        for (int j = 0; j < 3; j++){
            for (int ip = 0; ip < 4; ip++){
                xO[counter] = features_G4OH[ip][i][j];
                counter++;
            }
        }
    }

    for (int i = 0; i < noxygen; i++){
        for (int ip = 12; ip < nfeatures_O; ip++){
                 xO[i*nfeatures_O + ip] = (xO[i*nfeatures_O + ip] - wannier_GT_feature_scale[ip][0])/(wannier_GT_feature_scale[ip][2] - wannier_GT_feature_scale[ip][1]);
        }
    }

    auto options = torch::TensorOptions().dtype(torch::kFloat);
    torch::Tensor xO_tensor = torch::from_blob(xO.data(), {noxygen, nfeatures_O}, options);
    
    

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(xO_tensor);
    
    torch::Tensor yO = net.forward(inputs).toTensor();
    
    for (int i = 0; i < noxygen; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            for (int k = 0; k < 3; k++){
                wannier_mapped_rotated[i][j][k] = yO.index({i, j*3 + k}).item().toFloat() + wannier_GT_target_scale[j*3 + k];
            }
            
        }
    }

}

void backrotate_shift_reshape(vector<vector<vector<float> > > &wannier_mapped_rotated, vector<vector<float> > & wxyz, vector<vector<float> > & pos, vector<vector<vector<float> > > &rotamers){

    // reset 0
    for (int i=0; i<(4*noxygen); i++){
        for (int j = 0; j < 3; j++){
            wxyz[i][j] = 0;
        }
    }

    int nacount = 0;
    for (int i=0; i<noxygen; i++){
        for (int j=0; j<4; j++){
            for (int k=0; k<3; k++){
                for (int jj=0; jj<3; jj++){
                    wxyz[nacount][k] += wannier_mapped_rotated[i][j][jj] * rotamers[i][jj][k];
                }
                
            }
            nacount++;
            
        }
    }

    nacount = 0;
    for (int i=0; i<noxygen; i++){
        for (int j=0; j<4; j++){
            for (int k=0; k<3; k++){
                wxyz[nacount][k] += pos[i][k];
            }
            nacount++;
        }
    }

}

void backrotate_shift(vector<vector<vector<float> > > &wannier_mapped_rotated, vector<vector<vector<float> > > & wxyz, vector<vector<vector<float> > > &rotamers){

    for (int i=0; i<noxygen; i++){
        for (int j=0; j<4; j++){
            for (int k=0; k<3; k++){
                wxyz[i][j][k] = 0;
            }
        }
    }

    for (int i=0; i<noxygen; i++){
        for (int j=0; j<4; j++){
            for (int k=0; k<3; k++){
                for (int jj=0; jj<3; jj++){
                    wxyz[i][j][k] += wannier_mapped_rotated[i][j][jj] * rotamers[i][jj][k];
                }
                
            }            
        }
    }
}

vector<float> get_dipole(vector<vector<vector<float> > > &wxyz_mapped, vector<vector<vector<float> > > &vecrij_nnO){
    float qw = -2;
    float qH = 1;
    vector<float> dipole(3);
    for (int i=0; i<noxygen; i++){
        for (int j=0; j<4; j++){
            for (int k=0; k<3; k++){
                dipole[k] += qw * wxyz_mapped[i][j][k];
            }
        }
    }

    for (int i=0; i<noxygen; i++){
        for (int j=0; j<2; j++){
            for (int k=0; k<3; k++){
                dipole[k] += qH * vecrij_nnO[i][j][k];
            }
        }
    }

    return dipole;
}

void compute_wannier_peturb(float rij[][natoms], float pos_rotated[][natoms][3], vector<vector<vector<float> > > &wannier_peturb_mapped_rotated, float EO_rotated[], float EH_rotated[], vector<vector<float> > &parameters2_OO, vector<vector<float> > &parameters2_OH, vector<vector<float> > &parameters4_OO, vector<vector<float> > &parameters4_OH, torch::jit::script::Module &net){
    vector<vector<vector<float> > > features_G2OO(6, vector<vector<float> >(noxygen, vector<float> (3)));
    vector<vector<vector<float> > > features_G2OH(6, vector<vector<float> >(noxygen, vector<float> (3)));

    // get G2 features
    get_G2features_E(features_G2OO, parameters2_OO, 6, rij, 0, 0, EO_rotated);
    get_G2features_E(features_G2OH, parameters2_OH, 6, rij, 0, 1, EH_rotated);

    // assemble the features into desired shape
    int nfeatures_O = 6*3 + 6*3;
    vector<float> xO(noxygen * nfeatures_O);
    
    int counter = 0;
    for (int i = 0; i < noxygen; i++){
        
        for (int ip = 0; ip < 6; ip++){
            for (int j = 0; j < 3; j++){
                xO[counter] = features_G2OO[ip][i][j];
                counter++;
            }
        }
        
        for (int ip = 0; ip < 6; ip++){
            for (int j = 0; j < 3; j++){
                xO[counter] = features_G2OH[ip][i][j];
                counter++;
            }
        }
    }


    auto options = torch::TensorOptions().dtype(torch::kFloat);
    torch::Tensor xO_tensor = torch::from_blob(xO.data(), {noxygen, nfeatures_O}, options);
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(xO_tensor);
    
    torch::Tensor yO = net.forward(inputs).toTensor();

    for (int i = 0; i < noxygen; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            for (int k = 0; k < 3; k++){
                wannier_peturb_mapped_rotated[i][j][k] = yO.index({i, j*3 + k}).item().toFloat();
            }
            
        }
    }
}

void compute_force_peturbO(float rij[][natoms], float pos_rotated[][natoms][3], vector<vector<float> >  &force, float EO_rotated[], float EH_rotated[], vector<vector<float> > &parameters2_OO, vector<vector<float> > &parameters2_OH, vector<vector<float> > &parameters4_OO, vector<vector<float> > &parameters4_OH, torch::jit::script::Module &net){
    vector<vector<vector<float> > > features_G2OO(6, vector<vector<float> >(noxygen, vector<float> (3)));
    vector<vector<vector<float> > > features_G2OH(6, vector<vector<float> >(noxygen, vector<float> (3)));

    // get G2 features
    get_G2features_E(features_G2OO, parameters2_OO, 6, rij, 0, 0, EO_rotated);
    get_G2features_E(features_G2OH, parameters2_OH, 6, rij, 0, 1, EH_rotated);

    // assemble the features into desired shape
    int nfeatures_O = 6*3 + 6*3;
    vector<float> xO(noxygen * nfeatures_O);
    
    int counter = 0;
    for (int i = 0; i < noxygen; i++){
        
        for (int ip = 0; ip < 6; ip++){
            for (int j = 0; j < 3; j++){
                xO[counter] = features_G2OO[ip][i][j];
                counter++;
            }
        }
        
        for (int ip = 0; ip < 6; ip++){
            for (int j = 0; j < 3; j++){
                xO[counter] = features_G2OH[ip][i][j];
                counter++;
            }
        }
    }

    auto options = torch::TensorOptions().dtype(torch::kFloat);
    torch::Tensor xO_tensor = torch::from_blob(xO.data(), {noxygen, nfeatures_O}, options);
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(xO_tensor);
    
    torch::Tensor yO = net.forward(inputs).toTensor();

    for (int i = 0; i < noxygen; i++)
    {
       for (int k = 0; k < 3; k++){
            force[i][k] = yO.index({i, k}).item().toFloat();
       }      
    }
}

void compute_force_peturbH(float rij[][natoms], float pos_rotated[][natoms][3], vector<vector<float> >  &force, float EO_rotated[], float EH_rotated[], vector<vector<float> > &parameters2_HH, vector<vector<float> > &parameters2_HO, vector<vector<float> > &parameters4_HH, vector<vector<float> > &parameters4_HO, torch::jit::script::Module &net){
    vector<vector<vector<float> > > features_G2HH(6, vector<vector<float> >(nhydrogen, vector<float> (3)));
    vector<vector<vector<float> > > features_G2HO(6, vector<vector<float> >(nhydrogen, vector<float> (3)));

    // get G2 features
    get_G2features_E(features_G2HH, parameters2_HH, 6, rij, 1, 1, EH_rotated);
    get_G2features_E(features_G2HO, parameters2_HO, 6, rij, 1, 0, EO_rotated);

    // assemble the features into desired shape
    int nfeatures_O = 6*3 + 6*3;
    vector<float> xO(nhydrogen * nfeatures_O);
    
    int counter = 0;
    for (int i = 0; i < nhydrogen; i++){
        
        for (int ip = 0; ip < 6; ip++){
            for (int j = 0; j < 3; j++){
                xO[counter] = features_G2HH[ip][i][j];
                counter++;
            }
        }
        
        for (int ip = 0; ip < 6; ip++){
            for (int j = 0; j < 3; j++){
                xO[counter] = features_G2HO[ip][i][j];
                counter++;
            }
        }
    }

    auto options = torch::TensorOptions().dtype(torch::kFloat);
    torch::Tensor xO_tensor = torch::from_blob(xO.data(), {nhydrogen, nfeatures_O}, options);
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(xO_tensor);
    
    torch::Tensor yO = net.forward(inputs).toTensor();

    for (int i = 0; i < nhydrogen; i++)
    {
       for (int k = 0; k < 3; k++){
            force[i][k] = yO.index({i, k}).item().toFloat();
       }      
    }
}

void backrotate(vector<vector<float> > &f_rotated, vector<vector<float> > & f_original, vector<vector<vector<float> > > &rotamers){

    for (int i=0; i<f_original.size(); i++){
        for (int k=0; k<3; k++){
            f_original[i][k] = 0;
            for (int jj=0; jj<3; jj++){
                f_original[i][k] += f_rotated[i][jj] * rotamers[i][jj][k];
            }
                
        }
    }

}

void printE(vector<vector<float> > & EO, vector<vector<float> > & EH, int istep, int iter, ofstream &ff){
    // print data to file
    ff << istep << "  " << (iter+1) << "  " << 0 << " \n";
    for (int i = 0; i < noxygen; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            ff << EO[i][j] <<"  ";
        }
        ff<< endl;
    }
    for (int i = noxygen; i < natoms; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            ff << EH[i-noxygen][j] <<"  ";
        }
        ff<< endl;
    }
}

void print_general(vector<vector<float> > & wxyz, int istep, int iter, ofstream &ff){
    // print data to file
    ff << istep << "  " << (iter+1) << "  " << 0 << " \n";
    for (int i = 0; i < wxyz.size(); i++)
    {
        for (int j = 0; j < 3; j++)
        {
            ff << wxyz[i][j] <<"  ";
        }
        ff<< endl;
    }
    
}

float get_diff_copy(vector<vector<float> > & data_last, vector<vector<float> > & data){
    float diff = 0;
    for(int i = 0; i < data.size(); i++){
        for (int j = 0; j < data[0].size(); j++){
            diff += abs(data[i][j] - data_last[i][j]);
            data_last[i][j] = data[i][j];
        }
    }
    diff = diff / (data.size() * data[0].size());
    return diff;
}