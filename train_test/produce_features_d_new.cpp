// generate features and derivative of features using C file. don't need to utilize the broadcasting mechanism any more

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <string>
#include <vector>

#define natoms 192    // the total number of nucleus
#define noxygen 64    // the number of oxygen
#define nhydrogen 128 // the number of hydrogen

using namespace std;

double fc(double r)
{
    double rc = 12;
    double y = 0;
    if (r < rc)
    {
        y = pow(tanh(1 - r / rc), 3);
    }
    return y;
}

double dfc(double r)
{
    double rc = 12;
    double y = 0;
    if (r < rc)
    {
        y = -3 * pow(tanh(1 - r / rc), 2) / pow(cosh(1 - r / rc), 2) / rc;
    }
    return y;
}

double G2(double r12, double yeta, double rs)
{
    double y = exp(-yeta * (r12 - rs) * (r12 - rs)) * fc(r12);
    return y;
}

double dG2(double r12, double yeta, double rs) // this is only the radial part of dG2
{
    double y = -2 * yeta * (r12 - rs) * fc(r12) * exp(-yeta * pow((r12 - rs), 2)) + exp(-yeta * pow((r12 - rs), 2)) * dfc(r12);
    return y;
}

double dG4_ij(double r12, double r13, double r23, double cosalphaijk, double zeta, double yeta, double lam)
{
    double y = 0;
    double commonvalue = pow(2, 1 - zeta) * exp(-yeta * (r12 * r12 + r13 * r13 + r23 * r23)) * pow((1 + lam * cosalphaijk), (zeta-1)) ;
    y += zeta * lam * ( 1.0/ r13 - (r12*r12 + r13*r13 - r23*r23) / (2*r12*r12*r13))  * fc(r12) * fc(r13) *fc(r23);
    y += - 2 * r12 * yeta  * (1 + lam * cosalphaijk) * fc(r12) * fc(r13) *fc(r23);
    y += dfc(r12) * (1 + lam * cosalphaijk) * fc(r13) *fc(r23);
    y = y * commonvalue;
    return y;
}

vector<double> dG4_ij_ik_jk(double r12, double r13, double r23, double cosalphaijk, double zeta, double yeta, double lam)
{
    
    double fc12 = fc(r12);
    double fc13 = fc(r13);
    double fc23 = fc(r23);
    double dfc12 = dfc(r12);
    double dfc13 = dfc(r13);
    double dfc23 = dfc(r23);
    
    double commonvalue = pow(2, 1 - zeta) * exp(-yeta * (r12 * r12 + r13 * r13 + r23 * r23)) * pow((1 + lam * cosalphaijk), (zeta-1)) ;
    
    double y = 0;
    y += zeta * lam * ( 1.0/ r13 - (r12*r12 + r13*r13 - r23*r23) / (2*r12*r12*r13))  * fc12 * fc13 *fc23;
    y += - 2 * r12 * yeta  * (1 + lam * cosalphaijk) * fc12 * fc13 *fc23;
    y += dfc12 * (1 + lam * cosalphaijk) * fc13 *fc23;
    y = y * commonvalue;
    
    double y1 = 0;
    y1 += zeta * lam * ( 1.0/ r12 - (r12*r12 + r13*r13 - r23*r23) / (2*r13*r13*r12))  * fc12 * fc13 *fc23;
    y1 += - 2 * r13 * yeta  * (1 + lam * cosalphaijk) * fc12 * fc13 *fc23;
    y1 += dfc13 * (1 + lam * cosalphaijk) * fc12 *fc23;
    y1 = y1 * commonvalue;
    
    double y2 = 0;
    y2 += zeta * lam * (-r23/(r12 * r13)) * fc12 * fc13 *fc23;
    y2 += - 2 * r23 * yeta  * (1 + lam * cosalphaijk) * fc12 * fc13 *fc23;
    y2 += dfc23 * (1 + lam * cosalphaijk) * fc12 *fc13;
    y2 = y2 * commonvalue;
    
    vector<double> ys(3);
    ys[0]= y;
    ys[1] = y1;
    ys[2] = y2;
    return ys;
}

double dG4_jk(double r12, double r13, double r23, double cosalphaijk, double zeta, double yeta, double lam){
    double y = 0;
    double commonvalue = pow(2, 1 - zeta) * exp(-yeta * (r12 * r12 + r13 * r13 + r23 * r23)) * pow((1 + lam * cosalphaijk), (zeta-1)) ;
    y += zeta * lam * (-r23/(r12 * r13)) * fc(r12) * fc(r13) *fc(r23);
    y += - 2 * r23 * yeta  * (1 + lam * cosalphaijk) * fc(r12) * fc(r13) *fc(r23);
    y += dfc(r23) * (1 + lam * cosalphaijk) * fc(r12) *fc(r13);
    y = y * commonvalue;
    return y;
}

double G4(double r12, double r13, double r23, double cosalphaijk, double zeta, double yeta, double lam)
{
    double y = exp(-yeta * (r12 * r12 + r13 * r13 + r23 * r23)) * fc(r12) * fc(r13) * fc(r23) * pow((1 + lam * cosalphaijk), zeta);
    y = y * pow(2, 1 - zeta);
    return y;
}

void read_parameters(vector<vector<double> >& parameters, string fp_name, int nx)
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

void get_G2features(vector<vector<double> > &features, vector<vector<double> > &parameters, int nx, vector<vector<double> > &r, int id_i, int id_j, string fname)
{
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
            if (j != i)
            {

                for (int ip = 0; ip < nx; ip++)
                {
                    features[ip][i - id_i * noxygen] += G2(r[i][j], parameters[ip][1], parameters[ip][0]);
                }
            }
        }
    }

    ofstream fp(fname);
    for (int ip = 0; ip < nx; ip++)
    {
        for (int i = rangei[0]; i < rangei[1]; i++)
        {
            fp << features[ip][i - id_i * noxygen] << " ";
        }
        fp << "\n";
    }
    fp.close();
}

void get_dG2features(vector<vector<vector<vector<double> > > > &features, vector<vector<double> > &parameters, int nx, vector<vector<double> > &r, vector<vector<vector<double> > > &vecr, int id_i, int id_j, string fname)
{
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
            if (j != i)
            {
                for (int ip = 0; ip < nx; ip++)
                {
                    double dG2ij = dG2(r[i][j], parameters[ip][1], parameters[ip][0]);
                    for (int ix = 0; ix < 3; ix++)
                    {
                        features[ip][i - id_i * noxygen][j][ix] += dG2ij * vecr[i][j][ix] / r[i][j];
                        
                        features[ip][i - id_i * noxygen][i][ix] -= dG2ij * vecr[i][j][ix] / r[i][j];
                        
                    }
                }
            }
        }
    }
     
    
    
    ofstream fp(fname);
    for (int ip = 0; ip < nx; ip++)
    {
        for (int i = rangei[0]; i < rangei[1]; i++)
        {
            for (int j = 0; j < natoms; j++)
            {
                for (int ix = 0; ix < 3; ix++)
                {
                    fp << features[ip][i - id_i * noxygen][j][ix] << " ";
                }
            }
        }
    }
    fp << "\n";
    fp.close();
}

void get_G4features(vector<vector<double> > &features,  vector<vector<double> > &parameters, int nx, vector<vector<double> > &r, int id_i, int id_j, int id_k, string fname)
{
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

    for (int i = rangei[0]; i < rangei[1]; i++)
    {
        for (int j = rangej[0]; j < rangej[1]; j++)
        {
            if (j != i)
            {
                for (int k = rangek[0]; k < rangek[1]; k++)
                {
                    if ((k != j) && (k != i))
                    {
                        double cosijk = (r[i][j] * r[i][j] + r[i][k] * r[i][k] - r[j][k] * r[j][k]) / (2 * r[i][j] * r[i][k]);
                        for (int ip = 0; ip < nx; ip++)
                        {
                            features[ip][i - id_i * noxygen] += G4(r[i][j], r[i][k], r[j][k], cosijk, parameters[ip][3], parameters[ip][1], parameters[ip][2]);
                        }
                    }
                }
            }
        }
    }

    ofstream fp(fname);
    for (int ip = 0; ip < nx; ip++)
    {
        for (int i = rangei[0]; i < rangei[1]; i++)
        {
            fp << features[ip][i - id_i * noxygen] << " ";
        }
        fp << "\n";
    }
    fp.close();
}

void get_dG4features(vector<vector<vector<vector<double> > > > &features, vector<vector<double> > &parameters, int nx, vector<vector<double> > &r, vector<vector<vector<double> > > &vecr, int id_i, int id_j, int id_k, string fname)
{
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

    for (int i = rangei[0]; i < rangei[1]; i++)
    {
        for (int j = rangej[0]; j < rangej[1]; j++)
        {
            if (j != i)
            {
                for (int k = rangek[0]; k < rangek[1]; k++)
                {
                    if ((k != j) && (k != i))
                    {
                        double cosijk = (r[i][j] * r[i][j] + r[i][k] * r[i][k] - r[j][k] * r[j][k]) / (2 * r[i][j] * r[i][k]);
                        
                        for (int ip = 0; ip < nx; ip++)
                        {
                            vector<double> Gs = dG4_ij_ik_jk(r[i][j], r[i][k], r[j][k], cosijk, parameters[ip][3], parameters[ip][1], parameters[ip][2]);
                            double dG4_ij_ij = Gs[0];
                            double dG4_ij_ik = Gs[1];
                            double dG4_jk_jk = Gs[2];
                            for (int ix = 0; ix < 3; ix++)
                            {
                                features[ip][i - id_i * noxygen][j][ix] += dG4_ij_ij * vecr[i][j][ix] / r[i][j];
                                features[ip][i - id_i * noxygen][j][ix] += dG4_jk_jk * vecr[k][j][ix] / r[j][k];
                                
                                features[ip][i - id_i * noxygen][k][ix] += dG4_ij_ik * vecr[i][k][ix] / r[i][k];
                                features[ip][i - id_i * noxygen][k][ix] += dG4_jk_jk * vecr[j][k][ix] / r[k][j];
                                
                                features[ip][i - id_i * noxygen][i][ix] -= dG4_ij_ij * vecr[i][j][ix] / r[i][j];
                                features[ip][i - id_i * noxygen][i][ix] -= dG4_ij_ik * vecr[i][k][ix] / r[i][k];;
                            }
                        }
                    }
                }
            }
        }
    }
    ofstream fp(fname);
    for (int ip = 0; ip < nx; ip++)
    {
        for (int i = rangei[0]; i < rangei[1]; i++)
        {
            for (int j = 0; j < natoms; j++)
            {
                for (int ix = 0; ix < 3; ix++)
                {
                    fp << features[ip][i - id_i * noxygen][j][ix] << " ";
                }
            }
        }
    }
    fp << "\n";
    fp.close();
}

int main()
{

    vector<vector<double> > nxyz(natoms, vector<double>(3));
    vector<vector<double> > rij(natoms, vector<double>(natoms));       // get the distance matrix for the nucleus
    vector<vector<vector<double> > > vecrij(natoms, vector<vector<double> >(natoms, vector<double>(3))); // the vector between i and j
    double boxlength = 0;

    vector<vector<double> > parameters2_OO(8, vector<double>(4));
    vector<vector<double> > parameters2_OH(8, vector<double>(4));
    vector<vector<double> > parameters2_HO(8, vector<double>(4));
    vector<vector<double> > parameters2_HH(8, vector<double>(4));

    vector<vector<double> > parameters4_OOO(4, vector<double>(4));
    vector<vector<double> > parameters4_OOH(4, vector<double>(4));
    vector<vector<double> > parameters4_OHH(6, vector<double>(4));
    vector<vector<double> > parameters4_HHO(7, vector<double>(4));
    vector<vector<double> > parameters4_HOO(4, vector<double>(4));

    vector<vector<double> > features_G2OO(8, vector<double>(noxygen));
    vector<vector<double> > features_G2OH(8, vector<double>(noxygen));
    vector<vector<double> > features_G2HO(8, vector<double>(nhydrogen));
    vector<vector<double> > features_G2HH(8, vector<double>(nhydrogen));

    vector<vector<double> > features_G4OOO(4, vector<double>(noxygen));
    vector<vector<double> > features_G4OOH(4, vector<double>(noxygen));
    vector<vector<double> > features_G4OHH(6, vector<double>(noxygen));
    vector<vector<double> > features_G4HHO(7, vector<double>(nhydrogen));
    vector<vector<double> > features_G4HOO(4, vector<double>(nhydrogen));

    vector<vector<vector<vector<double> > > > features_dG2OO(8, vector<vector<vector<double> > >(noxygen, vector<vector<double> >(natoms, vector<double>(3))));
    vector<vector<vector<vector<double> > > > features_dG2OH(8, vector<vector<vector<double> > >(noxygen, vector<vector<double> >(natoms, vector<double>(3))));
    vector<vector<vector<vector<double> > > > features_dG2HO(8, vector<vector<vector<double> > >(nhydrogen, vector<vector<double> >(natoms, vector<double>(3))));
    vector<vector<vector<vector<double> > > > features_dG2HH(8, vector<vector<vector<double> > >(nhydrogen, vector<vector<double> >(natoms, vector<double>(3))));
    
    vector<vector<vector<vector<double> > > > features_dG4OOO(4, vector<vector<vector<double> > >(noxygen, vector<vector<double> >(natoms, vector<double>(3))));
    vector<vector<vector<vector<double> > > > features_dG4OOH(4, vector<vector<vector<double> > >(noxygen, vector<vector<double> >(natoms, vector<double>(3))));
    vector<vector<vector<vector<double> > > > features_dG4OHH(6, vector<vector<vector<double> > >(noxygen, vector<vector<double> >(natoms, vector<double>(3))));
    vector<vector<vector<vector<double> > > > features_dG4HHO(7, vector<vector<vector<double> > >(nhydrogen, vector<vector<double> >(natoms, vector<double>(3))));
    vector<vector<vector<vector<double> > > > features_dG4HOO(4, vector<vector<vector<double> > >(nhydrogen, vector<vector<double> >(natoms, vector<double>(3))));

    read_parameters(parameters2_OO, "G2_parameters_OO.txt", 8);
    read_parameters(parameters2_OH, "G2_parameters_OH.txt", 8);
    read_parameters(parameters2_HO, "G2_parameters_HO.txt", 8);
    read_parameters(parameters2_HH, "G2_parameters_HH.txt", 8);

    read_parameters(parameters4_OOO, "G4_parameters_OOO.txt", 4);
    read_parameters(parameters4_OOH, "G4_parameters_OOH.txt", 4);
    read_parameters(parameters4_OHH, "G4_parameters_OHH.txt", 6);
    read_parameters(parameters4_HHO, "G4_parameters_HHO.txt", 7);
    read_parameters(parameters4_HOO, "G4_parameters_HOO.txt", 4);

    for (int i = 0; i < natoms; i++)
    { //initialize rij
        for (int j = 0; j < natoms; j++)
        {
            rij[i][j] = 0;
        }
    }

    ifstream fbox("box.txt");
    fbox >> boxlength; // read in the boxlength of this configuration
    //cout << boxlength << "\n";
    fbox.close();

    ifstream fOxyz("Oxyz.txt"); // read in the xyz coordinates of the oxygen first
    ifstream fHxyz("Hxyz.txt"); // read in the xyz coordinates of the hydrogen next
    for (int i = 0; i < noxygen; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            fOxyz >> nxyz[i][j];
            //cout << nxyz[i][j] << "  ";
        }
        //cout << "\n";
    }
    for (int i = noxygen; i < (nhydrogen + noxygen); i++)
    {
        for (int j = 0; j < 3; j++)
        {
            fHxyz >> nxyz[i][j];
            //cout << nxyz[i][j] << "  ";
        }
        //cout << "\n";
    }

    // get rij and vecrij
    for (int i = 0; i < natoms; i++)
    {
        for (int j = 0; j < natoms; j++)
        {
            double disx = nxyz[j][0] - nxyz[i][0] - round((nxyz[j][0] - nxyz[i][0]) / boxlength) * boxlength;
            double disy = nxyz[j][1] - nxyz[i][1] - round((nxyz[j][1] - nxyz[i][1]) / boxlength) * boxlength;
            double disz = nxyz[j][2] - nxyz[i][2] - round((nxyz[j][2] - nxyz[i][2]) / boxlength) * boxlength;

            double dis = sqrt(disx * disx + disy * disy + disz * disz);
            rij[i][j] = dis;
            vecrij[i][j][0] = disx;
            vecrij[i][j][1] = disy;
            vecrij[i][j][2] = disz;
            // cout << vecrij[i][j] << "  ";
        }
        // cout << "\n";
    }

    // get G2 features
    get_G2features(features_G2OO, parameters2_OO, 8, rij, 0, 0, "features_G2OO.txt");
    get_G2features(features_G2OH, parameters2_OH, 8, rij, 0, 1, "features_G2OH.txt");
    get_G2features(features_G2HO, parameters2_HO, 8, rij, 1, 0, "features_G2HO.txt");
    get_G2features(features_G2HH, parameters2_HH, 8, rij, 1, 1, "features_G2HH.txt");

    // get G4 features
    get_G4features(features_G4OOO, parameters4_OOO, 4, rij, 0, 0, 0, "features_G4OOO.txt");
    get_G4features(features_G4OOH, parameters4_OOH, 4, rij, 0, 0, 1, "features_G4OOH.txt");
    get_G4features(features_G4OHH, parameters4_OHH, 6, rij, 0, 1, 1, "features_G4OHH.txt");
    get_G4features(features_G4HHO, parameters4_HHO, 7, rij, 1, 1, 0, "features_G4HHO.txt");
    get_G4features(features_G4HOO, parameters4_HOO, 4, rij, 1, 0, 0, "features_G4HOO.txt");

    
    // get dG2 features
    get_dG2features(features_dG2OO, parameters2_OO, 8, rij, vecrij, 0, 0, "features_dG2OO.txt");
    get_dG2features(features_dG2OH, parameters2_OH, 8, rij, vecrij, 0, 1, "features_dG2OH.txt");
    get_dG2features(features_dG2HO, parameters2_HO, 8, rij, vecrij, 1, 0, "features_dG2HO.txt");
    get_dG2features(features_dG2HH, parameters2_HH, 8, rij, vecrij, 1, 1, "features_dG2HH.txt");
    
    
    // get dG4 features
    get_dG4features(features_dG4OOO, parameters4_OOO, 4, rij, vecrij, 0, 0, 0,  "features_dG4OOO.txt");
    get_dG4features(features_dG4OOH, parameters4_OOH, 4, rij, vecrij, 0, 0, 1, "features_dG4OOH.txt");
    get_dG4features(features_dG4OHH, parameters4_OHH, 6, rij, vecrij, 0, 1, 1, "features_dG4OHH.txt");
    get_dG4features(features_dG4HHO, parameters4_HHO, 7, rij, vecrij, 1, 1, 0, "features_dG4HHO.txt");
    get_dG4features(features_dG4HOO, parameters4_HOO, 4, rij, vecrij, 1, 0, 0, "features_dG4HOO.txt");
    
    return 0;
}

