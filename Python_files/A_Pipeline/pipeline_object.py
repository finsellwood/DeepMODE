import pandas as pd
import numpy as np
import ROOT
from sklearn.datasets import load_diabetes
import root_numpy as rnp
import vector
import awkward as ak
import numba as nb
import tensorflow as tf
import sys

rootpath_load = "/vols/cms/dw515/outputs/SM/MPhysNtuples"
rootpath_save = "/vols/cms/fjo18/Masters2021"

variables_tt_1 = ["tauFlag_1",
    # Generator-level properties, actual decay mode of taus for training
    "pi_px_1", "pi_py_1", "pi_pz_1", "pi_E_1",
    "pi2_px_1", "pi2_py_1", "pi2_pz_1", "pi2_E_1",
    "pi3_px_1", "pi3_py_1", "pi3_pz_1", "pi3_E_1",
    # 4-momenta of the charged pions
    "pi0_px_1", "pi0_py_1", "pi0_pz_1", "pi0_E_1", 
    # 4-momenta of neutral pions
    "gam1_px_1", "gam1_py_1", "gam1_pz_1", "gam1_E_1",
    "gam2_px_1", "gam2_py_1", "gam2_pz_1", "gam2_E_1",
    # 4-momenta of two leading photons
    "gam_px_1", "gam_py_1", "gam_pz_1", "n_gammas_1",
    # 3-momenta vectors of all photons
    "sc1_px_1", "sc1_py_1", "sc1_pz_1", "sc1_E_1",
    # Shower-shape variables
    "sc1_r9_5x5_1", "sc1_ietaieta_5x5_1", #  "sc1_r9_1", "sc1_ietaieta_1",
    # 4-momentum of the supercluster
    "cl_px_1", "cl_py_1", "cl_pz_1", "sc1_Nclusters_1",
    # 3-momenta of clusters in supercluster
    "tau_px_1", "tau_py_1", "tau_pz_1", "tau_E_1",
    # 4-momenta of 'visible' tau
    "tau_decay_mode_1",
    #HPS algorithm decay mode
    "pt_1",
    ]
variables_tt_2 = ["tauFlag_2", 
    # Generator-level properties, actual decay mode of taus for training
    "pi_px_2", "pi_py_2", "pi_pz_2", "pi_E_2", 
    "pi2_px_2", "pi2_py_2", "pi2_pz_2", "pi2_E_2",
    "pi3_px_2", "pi3_py_2", "pi3_pz_2", "pi3_E_2",
    # 4-momenta of the charged pions
    # Note: pi2/pi3 only apply for 3pr modes
    "pi0_px_2", "pi0_py_2", "pi0_pz_2", "pi0_E_2", 
    # 4-momenta of neutral pions
    "gam1_px_2", "gam1_py_2", "gam1_pz_2", "gam1_E_2",
    "gam2_px_2", "gam2_py_2", "gam2_pz_2", "gam2_E_2",
    # 4-momenta of two leading photons
    "gam_px_2", "gam_py_2", "gam_pz_2", "n_gammas_2",
    # 3-momenta vectors of all photons
    "sc1_px_2", "sc1_py_2", "sc1_pz_2", "sc1_E_2",
    # Shower-shape variables
    "sc1_r9_5x5_2", "sc1_ietaieta_5x5_2", # "sc1_r9_2", "sc1_ietaieta_2",
    # 4-momentum of the supercluster
    "cl_px_2", "cl_py_2", "cl_pz_2", "sc1_Nclusters_2",
    # 3-momenta of clusters in supercluster
    "tau_px_2", "tau_py_2", "tau_pz_2", "tau_E_2",
    # 4-momenta of 'visible' tau
    "tau_decay_mode_2", 
    # HPS algorithm decay mode
    "pt_2",
    ]

gam1_2_4mom = ["gam1_E_2", "gam1_px_2", "gam1_py_2", "gam1_pz_2", ]
gam2_2_4mom = ["gam2_E_2", "gam2_px_2", "gam2_py_2", "gam2_pz_2", ]
pi0_2_4mom = ["pi0_E_2", "pi0_px_2", "pi0_py_2", "pi0_pz_2", ]
pi_2_4mom = ["pi_E_2", "pi_px_2", "pi_py_2", "pi_pz_2", ]
pi2_2_4mom = ["pi2_E_2", "pi2_px_2", "pi2_py_2", "pi2_pz_2", ]
pi3_2_4mom = ["pi3_E_2", "pi3_px_2", "pi3_py_2", "pi3_pz_2", ]
gam_2_3mom = ["gam_E_2", "gam_px_2", "gam_py_2", "gam_pz_2", ]
cl_2_3mom = ["cl_E_2", "cl_px_2", "cl_py_2", "cl_pz_2", ]
sc1_2_4mom = ["sc1_E_2", "sc1_px_2", "sc1_py_2", "sc1_pz_2", ]
tau_2_4mom = ["tau_E_2", "tau_px_2", "tau_py_2", "tau_pz_2", ]

measured4mom = [pi0_2_4mom, pi_2_4mom, pi2_2_4mom, \
pi3_2_4mom, sc1_2_4mom, gam_2_3mom, cl_2_3mom, ]
# Updated to include pi0 (previously excluded for some reason (13.12.21))
E_list = [a[0] for a in measured4mom]
px_list = [a[1] for a in measured4mom]
py_list = [a[2] for a in measured4mom]
pz_list = [a[3] for a in measured4mom]
fourmom_list = [E_list, px_list, py_list, pz_list]
# a list of actual columns with the lists of fourmomenta in
fourmom_list_colnames = ["E_full_list", "px_full_list", "py_full_list", "pz_full_list"]

pi_indices = np.array([0,1,1,1,0])
pi0_indices = np.array([1,0,0,0,0])
gamma_indices = np.array([0,0,0,0,0])
sc_indices = np.array([0,0,0,0,1])

class pipeline:
    def __init__(self, load_path, save_path):
        self.load_path = load_path
        self.save_path = save_path
        self.object_folder = "/A_Objects/Objects3_DM"
        # self.variables_tt_1 = ["tauFlag_1",
        #         # Generator-level properties, actual decay mode of taus for training
        #         "pi_px_1", "pi_py_1", "pi_pz_1", "pi_E_1",
        #         "pi2_px_1", "pi2_py_1", "pi2_pz_1", "pi2_E_1",
        #         "pi3_px_1", "pi3_py_1", "pi3_pz_1", "pi3_E_1",
        #         # 4-momenta of the charged pions
        #         "pi0_px_1", "pi0_py_1", "pi0_pz_1", "pi0_E_1", 
        #         # 4-momenta of neutral pions
        #         "gam1_px_1", "gam1_py_1", "gam1_pz_1", "gam1_E_1",
        #         "gam2_px_1", "gam2_py_1", "gam2_pz_1", "gam2_E_1",
        #         # 4-momenta of two leading photons
        #         "gam_px_1", "gam_py_1", "gam_pz_1", "n_gammas_1",
        #         # 3-momenta vectors of all photons
        #         "sc1_px_1", "sc1_py_1", "sc1_pz_1", "sc1_E_1",
        #         # Shower-shape variables
        #         "sc1_r9_5x5_1", "sc1_ietaieta_5x5_1", #  "sc1_r9_1", "sc1_ietaieta_1",
        #         # 4-momentum of the supercluster
        #         "cl_px_1", "cl_py_1", "cl_pz_1", "sc1_Nclusters_1",
        #         # 3-momenta of clusters in supercluster
        #         "tau_px_1", "tau_py_1", "tau_pz_1", "tau_E_1",
        #         # 4-momenta of 'visible' tau
        #         "tau_decay_mode_1",
        #         #HPS algorithm decay mode
        #         "pt_1",
        #        ]
        # self.variables_tt_2 = ["tauFlag_2", 
        #         # Generator-level properties, actual decay mode of taus for training
        #         "pi_px_2", "pi_py_2", "pi_pz_2", "pi_E_2", 
        #         "pi2_px_2", "pi2_py_2", "pi2_pz_2", "pi2_E_2",
        #         "pi3_px_2", "pi3_py_2", "pi3_pz_2", "pi3_E_2",
        #         # 4-momenta of the charged pions
        #         # Note: pi2/pi3 only apply for 3pr modes
        #         "pi0_px_2", "pi0_py_2", "pi0_pz_2", "pi0_E_2", 
        #         # 4-momenta of neutral pions
        #         "gam1_px_2", "gam1_py_2", "gam1_pz_2", "gam1_E_2",
        #         "gam2_px_2", "gam2_py_2", "gam2_pz_2", "gam2_E_2",
        #         # 4-momenta of two leading photons
        #         "gam_px_2", "gam_py_2", "gam_pz_2", "n_gammas_2",
        #         # 3-momenta vectors of all photons
        #         "sc1_px_2", "sc1_py_2", "sc1_pz_2", "sc1_E_2",
        #         # Shower-shape variables
        #         "sc1_r9_5x5_2", "sc1_ietaieta_5x5_2", # "sc1_r9_2", "sc1_ietaieta_2",
        #         # 4-momentum of the supercluster
        #         "cl_px_2", "cl_py_2", "cl_pz_2", "sc1_Nclusters_2",
        #         # 3-momenta of clusters in supercluster
        #         "tau_px_2", "tau_py_2", "tau_pz_2", "tau_E_2",
        #         # 4-momenta of 'visible' tau
        #         "tau_decay_mode_2", 
        #         # HPS algorithm decay mode
        #         "pt_2",
        #        ]

        # self.gam1_2_4mom = ["gam1_E_2", "gam1_px_2", "gam1_py_2", "gam1_pz_2", ]
        # self.gam2_2_4mom = ["gam2_E_2", "gam2_px_2", "gam2_py_2", "gam2_pz_2", ]
        # self.pi0_2_4mom = ["pi0_E_2", "pi0_px_2", "pi0_py_2", "pi0_pz_2", ]
        # self.pi_2_4mom = ["pi_E_2", "pi_px_2", "pi_py_2", "pi_pz_2", ]
        # self.pi2_2_4mom = ["pi2_E_2", "pi2_px_2", "pi2_py_2", "pi2_pz_2", ]
        # self.pi3_2_4mom = ["pi3_E_2", "pi3_px_2", "pi3_py_2", "pi3_pz_2", ]
        # self.gam_2_3mom = ["gam_E_2", "gam_px_2", "gam_py_2", "gam_pz_2", ]
        # self.cl_2_3mom = ["cl_E_2", "cl_px_2", "cl_py_2", "cl_pz_2", ]
        # self.sc1_2_4mom = ["sc1_E_2", "sc1_px_2", "sc1_py_2", "sc1_pz_2", ]
        # self.tau_2_4mom = ["tau_E_2", "tau_px_2", "tau_py_2", "tau_pz_2", ]

        # self.measured4mom = [self.pi0_2_4mom, self.pi_2_4mom, self.pi2_2_4mom, \
        #     self.pi3_2_4mom, self.sc1_2_4mom, self.gam_2_3mom, self.cl_2_3mom, ]
        # # Updated to include pi0 (previously excluded for some reason (13.12.21))
        # self.E_list = [a[0] for a in self.measured4mom]
        # self.px_list = [a[1] for a in self.measured4mom]
        # self.py_list = [a[2] for a in self.measured4mom]
        # self.pz_list = [a[3] for a in self.measured4mom]
        # self.fourmom_list = [self.E_list, self.px_list, self.py_list, self.pz_list]
        # # a list of actual columns with the lists of fourmomenta in
        # self.fourmom_list_colnames = ["E_full_list", "px_full_list", "py_full_list", "pz_full_list"]

    ### I/O COMMANDS ###
    def save_dataframe(self, dataframe, name):
        pd.to_pickle(dataframe, self.save_path + self.object_folder + name)
        #joblib.dump(dataframe, self.save_path + self.object_folder + name)

    def load_dataframe(self, loadpath, name):
        return pd.read_pickle(loadpath + name)

    def load_root_files(self):
        rootGG_tt = ROOT.TFile(self.load_path + "/MVAFILE_GluGluHToTauTauUncorrelatedDecay_Filtered_tt_2018.root")
        intreeGG_tt = rootGG_tt.Get("ntuple")
        rootVBF_tt = ROOT.TFile(self.load_path + "/MVAFILE_VBFHToTauTauUncorrelatedDecay_Filtered_tt_2018.root")
        intreeVBF_tt = rootVBF_tt.Get("ntuple")
        arrVBF_tt_1 = rnp.tree2array(intreeVBF_tt,branches=variables_tt_1)
        arrGG_tt_1 = rnp.tree2array(intreeGG_tt,branches=variables_tt_1)
        arrVBF_tt_2 = rnp.tree2array(intreeVBF_tt,branches=variables_tt_2)
        arrGG_tt_2 = rnp.tree2array(intreeGG_tt,branches=variables_tt_2)
        print("converting to dfs")
        del rootGG_tt, rootVBF_tt
        del intreeVBF_tt, intreeGG_tt
        dfVBF_tt_1 = pd.DataFrame(arrVBF_tt_1)
        dfGG_tt_1 = pd.DataFrame(arrGG_tt_1)
        dfVBF_tt_2 = pd.DataFrame(arrVBF_tt_2)
        dfGG_tt_2 = pd.DataFrame(arrGG_tt_2)
        del arrVBF_tt_1, arrGG_tt_1, arrVBF_tt_2, arrGG_tt_2
        print("reshaping")
        df_1 = pd.concat([dfVBF_tt_1,dfGG_tt_1], ignore_index=True) 
        df_2 = pd.concat([dfVBF_tt_2,dfGG_tt_2], ignore_index=True) 
        #combine gluon and vbf data for hadronic modes
        del dfVBF_tt_1, dfVBF_tt_2, dfGG_tt_2, dfGG_tt_1

        #~~ Separating the tt data into two separate datapoints ~~#

        df_1.set_axis(variables_tt_2, axis=1, inplace=True) 
        # rename axes to the same as variables 2
        self.df_full = pd.concat([df_1, df_2], ignore_index = True)
        del df_1, df_2
        #self.save_dataframe(self.df_full, "df_full.pkl")
    
    def load_hl_imvar(self, loadpath, hl_name, imvar_name):
        print("loading dataframes...")
        self.hl_df = self.load_dataframe(loadpath, hl_name)
        self.imvar_df = self.load_dataframe(loadpath, imvar_name)
        print("done")

    def split_full_by_dm(self, dataframe):
        # Dataframe must still have tauFlag_2 feature
        # Produces 6 dataframes for each mode, and a list of their lengths
        df_DM0 = dataframe[
            (dataframe["tauFlag_2"] == 0)
        ]
        lenDM0 = df_DM0.shape[0]

        df_DM1 = dataframe[
            (dataframe["tauFlag_2"] == 1)
        ]
        lenDM1 = df_DM1.shape[0]

        df_DM2 = dataframe[
            (dataframe["tauFlag_2"] == 2)
        ]
        lenDM2 = df_DM2.shape[0]

        df_DM10 = dataframe[
            (dataframe["tauFlag_2"] == 10)
        ]
        lenDM10 = df_DM10.shape[0]

        df_DM11 = dataframe[
            (dataframe["tauFlag_2"] == 11)
        ]
        lenDM11 = df_DM11.shape[0]

        df_DMminus1 = dataframe[
            (dataframe["tauFlag_2"] == -1)
        ]
        lenDMminus1 = df_DMminus1.shape[0]
        self.dm_lengths = [lenDM0, lenDM1, lenDM2, lenDM10, lenDM11, lenDMminus1]
        self.df_dm = [df_DM0, df_DM1, df_DM2, df_DM10, df_DM11, df_DMminus1]
        for a in range(len(self.df_dm)):
            self.save_dataframe(self.df_dm[a], "/df_dm%s.pkl" % a)
    
    def energyfinder(self, dataframe, momvariablenames_1):
        # This function generates the energy of the given massless object (magnitude of 3-mom) 
        # So that can be treated the same as the other four-momenta later
        arr = []
        counter = -1
        for i in dataframe[momvariablenames_1[1]]:
            counter += 1
            energ = []
            if i.size > 0:
                for j in range(len(i)):
                    momvect_j = [dataframe[momvariablenames_1[1]][counter][j],\
                                dataframe[momvariablenames_1[2]][counter][j],\
                                dataframe[momvariablenames_1[3]][counter][j]]
                    energ.append(np.sqrt(sum(x**2 for x in momvect_j)))
                arr.append(energ)
            else: arr.append([])
        dataframe[momvariablenames_1[0]] = arr

    def energyfinder_2(self, dataframe, momvariablenames_1):
        fourvect = vector.arr({"px": dataframe[momvariablenames_1[1]],\
                        "py": dataframe[momvariablenames_1[2]],\
                        "pz": dataframe[momvariablenames_1[3]],\
                        })
        dataframe[momvariablenames_1[0]] = ak.to_list(fourvect.mag)
            

    def calc_mass(self, dataframe, momvariablenames_1, momvariablenames_2 = None):
        # Also used for pi0 mass when momvariablenames_2 = None
        momvect1 = vector.obj(px = dataframe[momvariablenames_1[1]],\
                        py = dataframe[momvariablenames_1[2]],\
                        pz = dataframe[momvariablenames_1[3]],\
                        E = dataframe[momvariablenames_1[0]])
        if momvariablenames_2 is not None:
            momvect2 = vector.obj(px = dataframe[momvariablenames_2[1]],\
                            py = dataframe[momvariablenames_2[2]],\
                            pz = dataframe[momvariablenames_2[3]],\
                            E = dataframe[momvariablenames_2[0]])
            rho_vect = momvect1+momvect2
            name = "rho_mass"
        else:
            rho_vect = momvect1
            name = "pi0_2mass"
        dataframe[name] = rho_vect.mass

    def tau_eta(self, dataframe, momvariablenames_1):
        momvect1 = vector.obj(px = dataframe[momvariablenames_1[1]],\
                        py = dataframe[momvariablenames_1[2]],\
                        pz = dataframe[momvariablenames_1[3]],\
                        E = dataframe[momvariablenames_1[0]])
        dataframe["tau_eta"] = momvect1.eta  #tau eta (tau pt just a variable)

    def ang_var(self, dataframe, momvariablenames_1, momvariablenames_2, particlename): #same for gammas and pions
        momvect1 = vector.obj(px = dataframe[momvariablenames_1[1]],\
                        py = dataframe[momvariablenames_1[2]],\
                        pz = dataframe[momvariablenames_1[3]],\
                        E = dataframe[momvariablenames_1[0]])
        momvect2 = vector.obj(px = dataframe[momvariablenames_2[1]],\
                        py = dataframe[momvariablenames_2[2]],\
                        pz = dataframe[momvariablenames_2[3]],\
                        E = dataframe[momvariablenames_2[0]])
        
        diffphi = momvect1.phi - momvect2.phi
        diffeta = momvect1.eta - momvect2.eta
        diffr = np.sqrt(diffphi**2 + diffeta**2)
        Esum = dataframe[momvariablenames_1[0]] + dataframe[momvariablenames_2[0]]
        dataframe["delR_"+ particlename] = diffr
        dataframe["delPhi_"+ particlename] = diffphi
        dataframe["delEta_" + particlename] = diffeta
        dataframe["delR_xE_"+ particlename] = diffr * Esum
        dataframe["delPhi_xE_"+ particlename] = diffphi * Esum
        dataframe["delEta_xE_" + particlename] = diffeta * Esum

    def get_fourmomenta_lists(self, dataframe):
        for a in range(4):
            dataframe[fourmom_list_colnames[a]] = dataframe[fourmom_list[a][0]].apply(lambda x:\
                 np.array([x]).flatten().tolist())
            for b in range(1, len(fourmom_list[1])):   
                dataframe[fourmom_list_colnames[a]] += dataframe[fourmom_list[a][b]].apply(lambda x: \
                    np.array([x]).flatten().tolist())
      # This method means that the final lists are flat (i.e. clusters/photons dont have a nested structure)

    def phi_eta_find(self, dataframe):  
        # returns another dataframe
        self.get_fourmomenta_lists(dataframe)
        # Call the function for retrieving lists of 4-mom first
        output_dataframe = pd.DataFrame
        
        pi0vect = vector.obj(px = dataframe[pi0_2_4mom[1]],\
                            py = dataframe[pi0_2_4mom[2]],\
                            pz = dataframe[pi0_2_4mom[3]],\
                            E = dataframe[pi0_2_4mom[0]])

        pivect = vector.obj(px = dataframe[pi_2_4mom[1]],\
                            py = dataframe[pi_2_4mom[2]],\
                            pz = dataframe[pi_2_4mom[3]],\
                            E = dataframe[pi_2_4mom[0]])

        fourvect = vector.arr({"px": dataframe[fourmom_list_colnames[1]],\
                        "py": dataframe[fourmom_list_colnames[2]],\
                        "pz": dataframe[fourmom_list_colnames[3]],\
                            "E": dataframe[fourmom_list_colnames[0]]})
    
    
        tauvisfourvect = vector.obj(px = dataframe[tau_2_4mom[1]],\
                                py = dataframe[tau_2_4mom[2]],\
                                pz = dataframe[tau_2_4mom[3]],\
                                E = dataframe[tau_2_4mom[0]])

        pi0_phi = pi0vect.deltaphi(tauvisfourvect)
        pi0_eta = pi0vect.deltaeta(tauvisfourvect)
        pi_phi = pivect.deltaphi(tauvisfourvect)
        pi_eta = pivect.deltaphi(tauvisfourvect)

        pi0_transformed = vector.obj(px = pi0_phi, py = pi0_eta)
        pi_transformed = vector.obj(px = pi_phi, py = pi_eta)
        diff = pi_transformed - pi0_transformed

        unitvector = vector.obj(px = 1, py =0)
        rot_angles = ak.to_list(unitvector.deltaphi(diff))

        phis = ak.to_list(fourvect.deltaphi(tauvisfourvect))
        etas = ak.to_list(fourvect.deltaeta(tauvisfourvect))
        frac_energies = ak.to_list((fourvect.E/tauvisfourvect.E))
        frac_momenta = ak.to_list(fourvect.mag/tauvisfourvect.mag)

        output_dataframe = pd.DataFrame({'phis' : phis, 'etas' : etas, \
            'frac_energies' : frac_energies, 'frac_momenta' : frac_momenta, 'angles': rot_angles, "n_gammas_2": dataframe["n_gammas_2"]}) 
        # Added "n_gammas_2" column for proper parsing of photons and clusters

        return output_dataframe 
    
    def rot_x(self, x, y, theta):
        return np.array(x) * np.cos(theta) - np.array(y) * np.sin(theta)
    def rot_y(self, x, y, theta):
        return np.array(x) * np.sin(theta) + np.array(y) * np.cos(theta)
    
    def rotate_dataframe(self, dataframe):
        # rotates coords in a dataframe
        # Must have phis etas etc already
        dataframe["rot_phi"] = dataframe.apply(lambda x : self.rot_x(x["phis"], x["etas"], x["angles"]), axis = 1)
        dataframe["rot_eta"] = dataframe.apply(lambda x : self.rot_y(x["phis"], x["etas"], x["angles"]), axis = 1)
        cols_to_drop = ["phis", "etas", "angles"]
        dataframe.drop(columns = cols_to_drop, inplace = True)

    def drop_variables(self, dataframe):
        for a in measured4mom[4:]:
            dataframe.drop(columns = a, inplace = True)
        for a in measured4mom[:4]:
            dataframe.drop(columns = a[1:], inplace = True)
            # Keeps the energies for pi0, pi, pi2, pi3 as HL variables
        for a in fourmom_list_colnames:
            dataframe.drop(columns = a, inplace = True)

    def drop_variables_2(self, dataframe):
        # for pre-dataset drop
        dataframe.drop(["tauFlag_2", 
                  "gam1_px_2", "gam1_py_2", "gam1_pz_2", "gam1_E_2",
                  "gam2_px_2", "gam2_py_2", "gam2_pz_2", "gam2_E_2",
                  "tau_px_2", "tau_py_2", "tau_pz_2",
            ], axis=1, inplace = True)
        dataframe.reset_index(drop=True, inplace=True)
  
    def create_featuredesc(self, dataframe):
        # Creates feature descriptions for tfrecord datasets
        # dataframe is the HL data
        self.featurenames_hl = list(dataframe.columns)
        print(self.featurenames_hl)
        # npa = X_test.to_numpy()
        self.feature_description = {}
        self.fd_hl = {}
        self.fd_im_l = {}
        self.fd_im_s = {}
        for a in range(dataframe.shape[1]):
            self.feature_description[self.featurenames_hl[a]] = tf.io.FixedLenFeature([],tf.float32,default_value=0.0)
            self.fd_hl[self.featurenames_hl[a]] = tf.io.FixedLenFeature([],tf.float32,default_value=0.0)
        self.feature_description["large_image"] = tf.io.VarLenFeature(tf.int64)
        self.feature_description["small_image"] = tf.io.VarLenFeature(tf.int64)
        self.feature_description["flag"] =  tf.io.FixedLenFeature([],tf.int16)
        self.fd_im_l["large_image"] = tf.io.VarLenFeature(tf.int64)
        self.fd_im_s["small_image"] = tf.io.VarLenFeature(tf.int64)

    def create_featuredesc_flag(self):
        self.fd_flag = {}
        self.fd_flag["flag"] = tf.io.FixedLenFeature([],tf.int64)

    def generate_grids(self, row, dim1, dim2):
        halfdim1 = dim1/2
        halfdim2 = dim2/2

        # Two image grids (two vals per cell, [0] for energy and [1] for charge)
        phis = np.array(row["rot_phi"])
        etas = np.array(row["rot_eta"])
        energies = row["frac_energies"]
        momenta = row["frac_momenta"]
        photon_num = row["n_gammas_2"]

        phicoords =  [max(min(a, 20), 0) for a in np.floor((phis/1.21) * dim1 + halfdim1).astype(int)]
        etacoords =  [max(min(a, 20), 0) for a in np.floor(-1 * (etas/1.21) * dim1 + halfdim1).astype(int)]
        phicoords2 = [max(min(a, 10), 0) for a in np.floor((phis/0.2) * dim2 + halfdim2).astype(int)]
        etacoords2 = [max(min(a, 10), 0) for a in np.floor(-1 * (etas/0.2) * dim2 + halfdim2).astype(int)]
        
        int_energies = (np.minimum(np.abs(energies), 1) * 255).astype(np.uint8)
        int_momenta = (np.minimum(np.abs(momenta), 1) * 255).astype(np.uint8)

        real_mask = int_energies != 0
        masklen = len(real_mask)
    
        zerobuffer = np.zeros(masklen-5)
        gamma_buffer = np.append(np.ones(photon_num), np.zeros(masklen-5-photon_num))
        cluster_buffer = np.append(np.zeros(photon_num), np.ones(masklen-5-photon_num))
        pi_count = np.append(pi_indices, zerobuffer)
        pi0_count = np.append(pi0_indices, zerobuffer)
        sc_count = np.append(sc_indices, zerobuffer)
        gamma_count = np.append(gamma_indices, gamma_buffer)
        cluster_count = np.append(gamma_indices, cluster_buffer)
        layerlist = [int_energies, int_momenta, pi_count*real_mask, pi0_count*real_mask, sc_count*real_mask, gamma_count*real_mask,
        		cluster_count*real_mask,]
        no_layers = len(layerlist)
        grid1 = np.zeros((dim1,dim1, no_layers), np.uint8)
        grid2 = np.zeros((dim2,dim2, no_layers), np.uint8)
        
        for a in range(len(energies)):
            # if energies[a] != 0.0:
            for b in range(no_layers):
                grid1[etacoords[a],phicoords[a], b] += layerlist[b][a]
                # NOTE - if sum of elements exceeds 255 for a given cell then it will loop back to zero
                #if etacoords2[a] < dimension_s and etacoords2[a] >= 0 and phicoords2[a] < dimension_s and phicoords2[a] >=0:
                grid2[etacoords2[a],phicoords2[a], b] += layerlist[b][a]
                # Iterates through no_layers, so each layer has properties based on related layerlist component
        return grid1, grid2

    ### META COMMANDS ###

    def modify_dataframe(self, dataframe):
        print("energyfinder")
        self.energyfinder_2(dataframe, gam_2_3mom)
        self.energyfinder_2(dataframe, cl_2_3mom)
        print("calc mass")
        self.calc_mass(dataframe, pi0_2_4mom) # calc pi0 mass
        self.calc_mass(dataframe, pi_2_4mom, pi0_2_4mom) # calc rho mass
        print("frac features")
        dataframe["E_gam/E_tau"] = dataframe["gam1_E_2"].divide(dataframe["tau_E_2"]) #Egamma/Etau
        dataframe["E_pi/E_tau"] = dataframe["pi_E_2"].divide(dataframe["tau_E_2"]) #Epi/Etau
        dataframe["E_pi0/E_tau"] = dataframe["pi0_E_2"].divide(dataframe["tau_E_2"]) #Epi0/Etau
        print("ang features")
        self.tau_eta(dataframe, tau_2_4mom) # calc tau eta value
        self.ang_var(dataframe, gam1_2_4mom, gam2_2_4mom, "gam")
        self.ang_var(dataframe, pi0_2_4mom, pi_2_4mom, "pi")
        # NOTE THIS IS BETWEEN PI0 AND PI - IS THIS CORRECT?

    def create_imvar_dataframe(self, dataframe):
        print("generating coordinates")
        output_df = self.phi_eta_find(dataframe)
        print("rotating coordinates")
        self.rotate_dataframe(output_df)
        self.drop_variables(dataframe)
        return output_df

    def clear_dataframe(self):
        self.df_dm = None
    
    def generate_datasets(self, dataframe, imvar_dataframe, tfrecordpath, modeflag):        
        # needs to create the grid for each event, populate it, add to full tensor for event and save
        # per event
        # 0) drop unwanted columns from HL
        self.drop_variables_2(dataframe)
        # 1) create dictionaries for tfrecords
        self.create_featuredesc(dataframe)
        # 2) convert df to numpy, reset indices on imvar_df
        with tf.io.TFRecordWriter(tfrecordpath) as writer:
            npa = dataframe.to_numpy()
            imvar_dataframe.reset_index(drop=True, inplace=True)
            fulllen = imvar_dataframe.shape[0]
            del dataframe
            for a, row in imvar_dataframe.iterrows():
                #print(a/fulllen)
                event_dict = {}
                for b in range(npa.shape[1]):
                    event_dict[self.featurenames_hl[b]] = tf.train.Feature(float_list=\
                        tf.train.FloatList(value=[npa[a][b]]))
                # function for creating grids with a dataframe row
                (grid1, grid2) = self.generate_grids(row, 21, 11)

                event_dict["large_image"] = tf.train.Feature(int64_list=\
                    tf.train.Int64List(value=grid1.flatten()))
                event_dict["small_image"] = tf.train.Feature(int64_list=\
                    tf.train.Int64List(value=grid2.flatten()))
                event_dict["flag"] = tf.train.Feature(int64_list=\
                        tf.train.Int64List(value=[modeflag]))
                example = tf.train.Example(features=tf.train.Features(feature=event_dict))
                # print(example)
                writer.write(example.SerializeToString())
        # 3) iterate through events, using index of imvar_df to enumerate df
        # 4) create graphs, add HL + graphs to tfrecord and write
        print("done")

    def modify_by_decay_mode(self):
        for a in range(len(self.df_dm)):
            print(a)
            self.modify_dataframe(self.df_dm[a])
            imv_dm = self.create_imvar_dataframe(self.df_dm[a])
            self.save_dataframe(imv_dm, "/imvar_df_dm%s.pkl" % a)
            self.save_dataframe(self.df_dm[a], "/df_m_dm%s.pkl" % a)
            # New name to indicate modified
        self.clear_dataframe()

    def modify_single_df(self, filepath, name, index):
        df = self.load_dataframe(filepath, name)
        self.modify_dataframe(df)
        imv_dm = jez.create_imvar_dataframe(df)
        self.save_dataframe(imv_dm, "/imvar_df_dm%s.pkl" % index)
        self.save_dataframe(df, "/df_m_dm%s.pkl" % index)

        
        
# jez = pipeline(rootpath_load, rootpath_save)
# print(rootpath_save + jez.object_folder, "/ordereddf.pkl" )
# #jez.df_full = jez.load_dataframe(jez.save_path + jez.object_folder, "/ordereddf_modified.pkl")

# jez.load_root_files()
# jez.split_full_by_dm(jez.df_full)
# jez.modify_by_decay_mode()

# def modify_single_df(filepath, name, index):
#     df = jez.load_dataframe(filepath, name)
#     jez.modify_dataframe(df)
#     imv_dm = jez.create_imvar_dataframe(df)
#     jez.save_dataframe(imv_dm, "/imvar_df_dm%s.pkl" % index)
#     jez.save_dataframe(df, "/df_m_dm%s.pkl" % index)
# names = ["/df_dm0.pkl", "/df_dm1.pkl", "/df_dm2.pkl", "/df_dm3.pkl" \
#     "/df_dm4.pkl", "/df_dm5.pkl"]
names = ["/df_dm3.pkl", "/df_dm4.pkl", "/df_dm5.pkl"]
imvar_names = ["/imvar_df_dm0.pkl",  "/imvar_df_dm1.pkl", "/imvar_df_dm2.pkl", \
    "/imvar_df_dm3.pkl", "/imvar_df_dm4.pkl", "/imvar_df_dm5.pkl"]
df_mod_names = ["/df_m_dm0.pkl", "/df_m_dm1.pkl", "/df_m_dm2.pkl", \
    "/df_m_dm3.pkl", "/df_m_dm4.pkl", "/df_m_dm5.pkl"]
jez = pipeline(rootpath_load, rootpath_save)
index = int(sys.argv[1])
# Takes the index from an argument
print(index)
jez.load_hl_imvar(jez.save_path + jez.object_folder, df_mod_names[index], imvar_names[index])
jez.generate_datasets(jez.hl_df, jez.imvar_df, jez.save_path + "/E_TFRecords/dm%s.tfrecords" % index, index)
# def run_modifications(filepath):
#     for a in range(len(names)):
#         print(a)
#         jez.modify_single_df(filepath, names[a], a+3)
# run_modifications(jez.save_path + jez.object_folder)
# df = jez.load_dataframe(jez.save_path + jez.object_folder, "/df_dm0.pkl")
