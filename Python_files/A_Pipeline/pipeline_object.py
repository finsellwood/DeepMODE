import pandas as pd
import numpy as np
import ROOT
import root_numpy as rnp
import vector
import awkward as ak
import numba as nb
import tensorflow as tf
import sys

rootpath_load = "/vols/cms/dw515/outputs/SM/MPhysNtuples"
rootpath_save = "/vols/cms/fjo18/Masters2021"

# list of var names w.o. 'self.' is stored in 'notes.txt'

class feature_name_object:
    # A separate class with all the strings with feature names in
    def __init__(self) -> None:
        self.variables_tt_1 = ["tauFlag_1",
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
        self.variables_tt_2 = ["tauFlag_2", 
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

        self.gam1_2_4mom = ["gam1_E_2", "gam1_px_2", "gam1_py_2", "gam1_pz_2", ]
        self.gam2_2_4mom = ["gam2_E_2", "gam2_px_2", "gam2_py_2", "gam2_pz_2", ]
        self.pi0_2_4mom = ["pi0_E_2", "pi0_px_2", "pi0_py_2", "pi0_pz_2", ]
        self.pi_2_4mom = ["pi_E_2", "pi_px_2", "pi_py_2", "pi_pz_2", ]
        self.pi2_2_4mom = ["pi2_E_2", "pi2_px_2", "pi2_py_2", "pi2_pz_2", ]
        self.pi3_2_4mom = ["pi3_E_2", "pi3_px_2", "pi3_py_2", "pi3_pz_2", ]
        self.gam_2_3mom = ["gam_E_2", "gam_px_2", "gam_py_2", "gam_pz_2", ]
        self.cl_2_3mom = ["cl_E_2", "cl_px_2", "cl_py_2", "cl_pz_2", ]
        self.sc1_2_4mom = ["sc1_E_2", "sc1_px_2", "sc1_py_2", "sc1_pz_2", ]
        self.tau_2_4mom = ["tau_E_2", "tau_px_2", "tau_py_2", "tau_pz_2", ]

        self.measured4mom = [self.pi0_2_4mom, self.pi_2_4mom, self.pi2_2_4mom, \
            self.pi3_2_4mom, self.sc1_2_4mom, self.gam_2_3mom, self.cl_2_3mom, ]
        # Updated to include pi0 (previously excluded for some reason (13.12.21))
        self.E_list = [a[0] for a in self.measured4mom]
        self.px_list = [a[1] for a in self.measured4mom]
        self.py_list = [a[2] for a in self.measured4mom]
        self.pz_list = [a[3] for a in self.measured4mom]
        self.fourmom_list = [self.E_list, self.px_list, self.py_list, self.pz_list]
        # a list of actual columns with the lists of fourmomenta in
        self.fourmom_list_colnames = ["E_full_list", "px_full_list", "py_full_list", "pz_full_list"]
        
        self.pi_indices = np.array([0,1,1,1,0])
        self.pi0_indices = np.array([1,0,0,0,0])
        self.gamma_indices = np.array([0,0,0,0,0])
        self.sc_indices = np.array([0,0,0,0,1])

class pipeline(feature_name_object):
    def __init__(self, load_path, save_path):
        self.load_path = load_path
        self.save_path = save_path
        self.object_folder = "/A_Objects/Objects3_DM"
        feature_name_object.__init__(self)

    ### I/O COMMANDS ###
    def save_dataframe(self, dataframe, name):
        pd.to_pickle(dataframe, self.save_path + self.object_folder + name)
        #joblib.dump(dataframe, self.save_path + self.object_folder + name)

    def load_dataframe(self, loadpath, name):
        return pd.read_pickle(loadpath + name)

    def load_root_files(self, file_name):
        rootGG_tt = ROOT.TFile(self.load_path + file_name)
        intreeGG_tt = rootGG_tt.Get("ntuple")
        rootVBF_tt = ROOT.TFile(self.load_path + file_name)
        intreeVBF_tt = rootVBF_tt.Get("ntuple")
        arrVBF_tt_1 = rnp.tree2array(intreeVBF_tt,branches=self.variables_tt_1)
        arrGG_tt_1 = rnp.tree2array(intreeGG_tt,branches=self.variables_tt_1)
        arrVBF_tt_2 = rnp.tree2array(intreeVBF_tt,branches=self.variables_tt_2)
        arrGG_tt_2 = rnp.tree2array(intreeGG_tt,branches=self.variables_tt_2)
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

        df_1.set_axis(self.variables_tt_2, axis=1, inplace=True) 
        # rename axes to the same as variables 2
        self.df_full = pd.concat([df_1, df_2], ignore_index = True)
        #self.save_dataframe(self.df_full, "df_full.pkl")

    def load_root_files_2(self, which, file_name):
        root = ROOT.TFile(self.load_path + "/" + file_name)
        intree = root.Get("ntuple")
        if which == 1:
            arr = rnp.tree2array(intree,branches=self.variables_tt_1)
        elif which == 2:
            arr = rnp.tree2array(intree,branches=self.variables_tt_2)
        else: raise Exception("Incorrect tau label: can be either 1 or 2")
        print("converting to dfs")
        del root
        del intree
        df = pd.DataFrame(arr)
        del arr
        print("reshaping")
        if which == 1:
            df.set_axis(self.variables_tt_2, axis=1, inplace=True)
        # rename axes to the same as variables 2
        self.df_full = df
        self.decay_mode = df["tau_decay_mode_2"]
        del df
        
    def load_single_event(self, event, which, file_name):
        #file should not be hardcoded - pass from yaml
        file_ = ROOT.TFile(self.load_path +"/"+ file_name)
        tree = file_.Get("ntuple") #do GluGlu and VBF separately
        tree.GetEntry(event)
        arr = {}
        self.df_full = pd.Series(dtype=float)
        if which == 1:
            for i in self.variables_tt_1:
                #arr[i] = np.array(getattr(tree, i))
                # self.df_full[i] = [np.array(getattr(tree, i))]
                # self.df_full[i] = np.asarray(getattr(tree, i)).astype(float)
                arr[i] = np.asarray(getattr(tree, i)).astype(float)
        elif which == 2:
            for i in self.variables_tt_2:
                #arr[i] = np.array(getattr(tree, i))
                # self.df_full[i] = [np.array(getattr(tree, i))]
                # self.df_full[i] = np.asarray(getattr(tree, i)).astype(float)
                arr[i] = np.asarray(getattr(tree, i)).astype(float)
        else: raise Exception("Incorrect tau label: can be either 1 or 2") 
        # print("converting to dfs")
        # print(self.df_full["gam_px_1"])
        # print(self.df_full.head())
        #df = pd.DataFrame(arr, index[0])
        # rename axes to the same as variables 2
        if which == 1: 
            # self.df_full.set_axis(self.variables_tt_2, axis=1, inplace=True)
            # self.df_full.set_axis(self.variables_tt_2, inplace=True)
            for a in range(len(self.variables_tt_2)):
                arr[self.variables_tt_2[a]] = arr.pop(self.variables_tt_1[a])
        else: 
            pass
        # print(self.df_full[self.pi0_2_4mom])
        # self.df_full = self.df_full.to_frame().T
        # print(self.df_full.head())
        self.df_full = arr
        
    def load_single_event2(self, event):
        #file should not be hardcoded - pass from yaml
        file_ = ROOT.TFile(self.load_path + "/MVAFILE_GluGluHToTauTauUncorrelatedDecay_Filtered_tt_2018.root")
        tree = file_.Get("ntuple") #do GluGlu and VBF separately
        tree.GetEntry(event)
        arr1 = {}
        arr2 = {}
        self.df_full = pd.Series(dtype=float)
        for i in self.variables_tt_1:
            #arr[i] = np.array(getattr(tree, i))
            # self.df_full[i] = [np.array(getattr(tree, i))]
            # self.df_full[i] = np.asarray(getattr(tree, i)).astype(float)
            arr1[i] = np.asarray(getattr(tree, i)).astype(float)
        for i in self.variables_tt_2:
            #arr[i] = np.array(getattr(tree, i))
            # self.df_full[i] = [np.array(getattr(tree, i))]
            # self.df_full[i] = np.asarray(getattr(tree, i)).astype(float)
            arr2[i] = np.asarray(getattr(tree, i)).astype(float)
        # print(self.df_full["gam_px_1"])
        # print(self.df_full.head())
        #df = pd.DataFrame(arr, index[0])
        # rename axes to the same as variables 2
        for a in range(len(self.variables_tt_2)):
            arr1[self.variables_tt_2[a]] = arr1.pop(self.variables_tt_1[a])
        else: 
            pass
        df = pd.DataFrame([arr1,arr2])
        # print(self.df_full[self.pi0_2_4mom])
        # self.df_full = self.df_full.to_frame().T
        # print(self.df_full.head())
        self.df_full = df
        
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
        # print(momvect1)
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
        # print(rho_vect)
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
            dataframe[self.fourmom_list_colnames[a]] = dataframe[self.fourmom_list[a][0]].apply(lambda x:\
                 np.array([x]).flatten().tolist())
            for b in range(1, len(self.fourmom_list[1])):   
                dataframe[self.fourmom_list_colnames[a]] += dataframe[self.fourmom_list[a][b]].apply(lambda x: \
                    np.array([x]).flatten().tolist())

    def get_fourmomenta_lists_se(self, dataframe):
        for a in range(4):
            dataframe[self.fourmom_list_colnames[a]] = np.array([dataframe[self.fourmom_list[a][0]]]).flatten().tolist()
            for b in range(1, len(self.fourmom_list[1])):   
                dataframe[self.fourmom_list_colnames[a]] += np.array([dataframe[self.fourmom_list[a][b]]]).flatten().tolist()
      # This method means that the final lists are flat (i.e. clusters/photons dont have a nested structure)

    def phi_eta_find(self, dataframe, one_event = False):  
        # returns another dataframe
        if one_event:
            self.get_fourmomenta_lists_se(dataframe)
        else:    
            self.get_fourmomenta_lists(dataframe)
        # Call the function for retrieving lists of 4-mom first
        
        pi0vect = vector.obj(px = dataframe[self.pi0_2_4mom[1]],\
                            py = dataframe[self.pi0_2_4mom[2]],\
                            pz = dataframe[self.pi0_2_4mom[3]],\
                            E = dataframe[self.pi0_2_4mom[0]])

        pivect = vector.obj(px = dataframe[self.pi_2_4mom[1]],\
                            py = dataframe[self.pi_2_4mom[2]],\
                            pz = dataframe[self.pi_2_4mom[3]],\
                            E = dataframe[self.pi_2_4mom[0]])

        fourvect = vector.arr({"px": dataframe[self.fourmom_list_colnames[1]],\
                        "py": dataframe[self.fourmom_list_colnames[2]],\
                        "pz": dataframe[self.fourmom_list_colnames[3]],\
                            "E": dataframe[self.fourmom_list_colnames[0]]})
    
    
        tauvisfourvect = vector.obj(px = dataframe[self.tau_2_4mom[1]],\
                                py = dataframe[self.tau_2_4mom[2]],\
                                pz = dataframe[self.tau_2_4mom[3]],\
                                E = dataframe[self.tau_2_4mom[0]])

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
        for a in self.measured4mom[4:]:
            dataframe.drop(columns = a, inplace = True)
        for a in self.measured4mom[:4]:
            dataframe.drop(columns = a[1:], inplace = True)
            # Keeps the energies for pi0, pi, pi2, pi3 as HL variables
        for a in self.fourmom_list_colnames:
            dataframe.drop(columns = a, inplace = True)

    def drop_variables_se(self, dictionary):
        for a in self.measured4mom[4:]:
            for b in a:
                del dictionary[b]
            # dictionary.pop(a)
        for a in self.measured4mom[:4]:
            for b in a[1:]:
            # dictionary.pop(a[1:])
                del dictionary[b]
            # Keeps the energies for pi0, pi, pi2, pi3 as HL variables
        for a in self.fourmom_list_colnames:
            del dictionary[a]
            # dictionary.pop(a)

    def drop_variables_2(self, dataframe):
        # for pre-dataset drop
        dataframe.drop(["tauFlag_2", 
                  "gam1_px_2", "gam1_py_2", "gam1_pz_2", "gam1_E_2",
                  "gam2_px_2", "gam2_py_2", "gam2_pz_2", "gam2_E_2",
                  "tau_px_2", "tau_py_2", "tau_pz_2",
            ], axis=1, inplace = True)
        dataframe.reset_index(drop=True, inplace=True)
    
    def drop_variables_2_se(self,dictionary):
        to_drop = ["tauFlag_2", 
                  "gam1_px_2", "gam1_py_2", "gam1_pz_2", "gam1_E_2",
                  "gam2_px_2", "gam2_py_2", "gam2_pz_2", "gam2_E_2",
                  "tau_px_2", "tau_py_2", "tau_pz_2",]
        for a in to_drop:
            del dictionary[a]
  
    def create_featuredesc(self, dataframe):
        # Creates feature descriptions for tfrecord datasets
        # Feature descriptions are dictionaries describing the structure of the data within a given feature
        # Necessary for loading and saving tensorflow datasets in the tfrecord format
        # 'dataframe' is the HL data
        self.featurenames_hl = list(dataframe.columns)
        print(self.featurenames_hl)
        self.feature_description = {}
        self.fd_hl = {}
        self.fd_im_l = {}
        self.fd_im_s = {}
        self.fd_flag = {}
        for a in range(dataframe.shape[1]):
            self.feature_description[self.featurenames_hl[a]] = tf.io.FixedLenFeature([],tf.float32,default_value=0.0)
            self.fd_hl[self.featurenames_hl[a]] = tf.io.FixedLenFeature([],tf.float32,default_value=0.0)
        self.feature_description["large_image"] = tf.io.VarLenFeature(tf.int64)
        self.feature_description["small_image"] = tf.io.VarLenFeature(tf.int64)
        self.feature_description["flag"] =  tf.io.FixedLenFeature([],tf.int16)
        self.fd_im_l["large_image"] = tf.io.VarLenFeature(tf.int64)
        self.fd_im_s["small_image"] = tf.io.VarLenFeature(tf.int64)
        self.fd_flag["flag"] = tf.io.FixedLenFeature([6],tf.int64)

    def create_featuredesc2(self, dataframe):
        # Creates feature descriptions for tfrecord datasets
        # This version is for the three-feature tfrecords which might work where the 28 feature ones didnt
        self.feature_description = {}
        self.fd_hl = {}
        self.fd_im_l = {}
        self.fd_im_s = {}
        self.fd_flag = {}
        self.hl_no_features = dataframe.shape[-1]

        self.feature_description["hl"] = tf.io.FixedLenFeature([self.hl_no_features],tf.float32)
        self.feature_description["large_image"] = tf.io.FixedLenFeature([21,21,7],tf.int64)
        self.feature_description["small_image"] = tf.io.FixedLenFeature([11,11,7],tf.int64)
        self.fd_im_l["large_image"] = tf.io.FixedLenFeature([21,21,7],tf.int64)
        self.fd_im_s["small_image"] = tf.io.FixedLenFeature([11,11,7],tf.int64)
        self.fd_hl["hl"] = tf.io.FixedLenFeature([self.hl_no_features],tf.float32)
        self.fd_flag["flag"] = tf.io.FixedLenFeature([6],tf.int64)


    def calc_no_events(self, dataframe):
        return dataframe.shape[0]

    def generate_grids(self, row, dim1, dim2):
        halfdim1 = dim1/2
        halfdim2 = dim2/2

        # Two image grids (two vals per cell, [0] for energy and [1] for charge)
        phis = np.array(row["rot_phi"])
        etas = np.array(row["rot_eta"])
        energies = row["frac_energies"]
        momenta = row["frac_momenta"]
        photon_num = int(row["n_gammas_2"]) #single_event anal addition, hopefully doesn't break

        phicoords =  [max(min(a, 20), 0) for a in np.floor((phis/1.21) * dim1 + halfdim1).astype(int)]
        etacoords =  [max(min(a, 20), 0) for a in np.floor(-1 * (etas/1.21) * dim1 + halfdim1).astype(int)]
        phicoords2 = [max(min(a, 10), 0) for a in np.floor((phis/0.2) * dim2 + halfdim2).astype(int)]
        etacoords2 = [max(min(a, 10), 0) for a in np.floor(-1 * (etas/0.2) * dim2 + halfdim2).astype(int)]
        
        int_energies = (np.minimum(np.abs(energies), 1) * 255).astype(np.uint16)
        int_momenta = (np.minimum(np.abs(momenta), 1) * 255).astype(np.uint16)

        real_mask = int_energies != 0
        masklen = len(real_mask)
    
        zerobuffer = np.zeros(masklen-5)
        gamma_buffer = np.append(np.ones(photon_num), np.zeros(masklen-5-photon_num))
        cluster_buffer = np.append(np.zeros(photon_num), np.ones(masklen-5-photon_num))
        pi_count = np.append(self.pi_indices, zerobuffer)
        pi0_count = np.append(self.pi0_indices, zerobuffer)
        sc_count = np.append(self.sc_indices, zerobuffer)
        gamma_count = np.append(self.gamma_indices, gamma_buffer)
        cluster_count = np.append(self.gamma_indices, cluster_buffer)
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
        self.energyfinder_2(dataframe, self.gam_2_3mom)
        self.energyfinder_2(dataframe, self.cl_2_3mom)
        print("calc mass")
        self.calc_mass(dataframe, self.pi0_2_4mom) # calc pi0 mass
        self.calc_mass(dataframe, self.pi_2_4mom, self.pi0_2_4mom) # calc rho mass
        print("frac features")
        dataframe["E_gam/E_tau"] = dataframe["gam1_E_2"].divide(dataframe["tau_E_2"]) #Egamma/Etau
        dataframe["E_pi/E_tau"] = dataframe["pi_E_2"].divide(dataframe["tau_E_2"]) #Epi/Etau
        dataframe["E_pi0/E_tau"] = dataframe["pi0_E_2"].divide(dataframe["tau_E_2"]) #Epi0/Etau
        print("ang features")
        self.tau_eta(dataframe, self.tau_2_4mom) # calc tau eta value
        self.ang_var(dataframe, self.gam1_2_4mom, self.gam2_2_4mom, "gam")
        self.ang_var(dataframe, self.pi0_2_4mom, self.pi_2_4mom, "pi")
        # NOTE THIS IS BETWEEN PI0 AND PI - IS THIS CORRECT?

    def modify_dataframe_se(self, dataframe):
        # ACCEPTS DICTIONARY NOT DATAFRAME
        # print("energyfinder")
        self.energyfinder_2(dataframe, self.gam_2_3mom)
        self.energyfinder_2(dataframe, self.cl_2_3mom)
        # print("calc mass")
        self.calc_mass(dataframe, self.pi0_2_4mom) # calc pi0 mass
        self.calc_mass(dataframe, self.pi_2_4mom, self.pi0_2_4mom) # calc rho mass
        # print("frac features")
        dataframe["E_gam/E_tau"] = dataframe["gam1_E_2"]/dataframe["tau_E_2"] #Egamma/Etau
        dataframe["E_pi/E_tau"] = dataframe["pi_E_2"]/dataframe["tau_E_2"] #Epi/Etau
        dataframe["E_pi0/E_tau"] = dataframe["pi0_E_2"]/dataframe["tau_E_2"] #Epi0/Etau
        # print("ang features")
        # self.df_full = pd.DataFrame.from_dict(self.df_full, orient="index").T
        self.tau_eta(dataframe, self.tau_2_4mom) # calc tau eta value
        self.ang_var(dataframe, self.gam1_2_4mom, self.gam2_2_4mom, "gam")
        self.ang_var(dataframe, self.pi0_2_4mom, self.pi_2_4mom, "pi")
        # NOTE THIS IS BETWEEN PI0 AND PI - IS THIS CORRECT?

    def create_imvar_dataframe(self, dataframe, one_event = False):
        # print("generating coordinates")
        output_df = self.phi_eta_find(dataframe, one_event)
        # print("rotating coordinates")
        self.rotate_dataframe(output_df)
        if one_event:
            self.drop_variables_se(dataframe)
        else:
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
        path = tfrecordpath + "/dm%s.tfrecords" % index
        length = self.calc_no_events(dataframe)
        # 2) convert df to numpy, reset indices on imvar_df
        with tf.io.TFRecordWriter(path) as writer:
            npa = dataframe.to_numpy()
            imvar_dataframe.reset_index(drop=True, inplace=True)
            fulllen = imvar_dataframe.shape[0]
            del dataframe
            for a, row in imvar_dataframe.iterrows():
                print(a/fulllen)
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

        with open(tfrecordpath + '/dm%s_%s.txt' % (index, length), 'w') as f:
            f.write(str(length))
        print("done")

    def generate_datasets2(self, dataframe, imvar_dataframe, tfrecordpath, modeflag):        
        # needs to create the grid for each event, populate it, add to full tensor for event and save
        # per event
        onehot_flag = [0,0,0,0,0,0]
        onehot_flag[modeflag] = 1
        # 0) drop unwanted columns from HL
        self.drop_variables_2(dataframe)
        # 1) create dictionaries for tfrecords
        self.create_featuredesc2(dataframe)
        path = tfrecordpath + "/dm%s_3in.tfrecords" % index
        length = self.calc_no_events(dataframe)
        # 2) convert df to numpy, reset indices on imvar_df
        with tf.io.TFRecordWriter(path) as writer:
            print('Writing')
            npa = dataframe.to_numpy()
            imvar_dataframe.reset_index(drop=True, inplace=True)
            fulllen = imvar_dataframe.shape[0]
            del dataframe
            for a, row in imvar_dataframe.iterrows():
                print(a/fulllen)
                event_dict = {}
                event_dict["hl"] = tf.train.Feature(float_list=\
                    tf.train.FloatList(value=npa[a].flatten()))
                # function for creating grids with a dataframe row
                (grid1, grid2) = self.generate_grids(row, 21, 11)

                event_dict["large_image"] = tf.train.Feature(int64_list=\
                    tf.train.Int64List(value=grid1.flatten()))
                event_dict["small_image"] = tf.train.Feature(int64_list=\
                    tf.train.Int64List(value=grid2.flatten()))
                event_dict["Outputs"] = tf.train.Feature(int64_list=\
                        tf.train.Int64List(value=onehot_flag))
                example = tf.train.Example(features=tf.train.Features(feature=event_dict))
                # print(example)
                writer.write(example.SerializeToString())
        print("done writing, saving length")
        with open(tfrecordpath + '/dm%s_%s.txt' % (index, length), 'w') as f:
            f.write(str(length))
        print("done")

    def generate_datasets_anal(self, dataframe, imvar_dataframe, tfrecordpath):        
        # needs to create the grid for each event, populate it, add to full tensor for event and save
        # per event
        # 0) drop unwanted columns from HL
        self.drop_variables_2_se(dataframe)
        # 1) create dictionaries for tfrecords
        # self.create_featuredesc2(dataframe)
        path = tfrecordpath + "/dm.tfrecords"
        length = 1 # self.calc_no_events(dataframe)
        dataframe = pd.DataFrame(dataframe, index=[0])
        # 2) convert df to numpy, reset indices on imvar_df
        with tf.io.TFRecordWriter(path) as writer:
            # print('Writing')
            npa = dataframe.to_numpy()
            imvar_dataframe.reset_index(drop=True, inplace=True)
            fulllen = imvar_dataframe.shape[0]
            del dataframe
            imvar_dataframe = (imvar_dataframe.groupby(["n_gammas_2"]\
                ).agg({'frac_energies': lambda x: x.tolist(),\
                'frac_momenta': lambda x: x.tolist(),'rot_phi':\
                    lambda x: x.tolist(),'rot_eta': lambda x: \
                        x.tolist()}).reset_index())
            for a, row in imvar_dataframe.iterrows():
                # print(a/fulllen)
                event_dict = {}
                event_dict["hl"] = tf.train.Feature(float_list=\
                    tf.train.FloatList(value=npa[a].flatten()))
                # print(event_dict)
                # function for creating grids with a dataframe row
                (grid1, grid2) = self.generate_grids(row, 21, 11)

                event_dict["large_image"] = tf.train.Feature(int64_list=\
                    tf.train.Int64List(value=grid1.flatten()))
                event_dict["small_image"] = tf.train.Feature(int64_list=\
                    tf.train.Int64List(value=grid2.flatten()))
                example = tf.train.Example(features=tf.train.Features(feature=event_dict))
                # print(example)
                writer.write(example.SerializeToString())
        # print("done")
    

    def generate_datasets_anal_2(self, dataframe, imvar_dataframe, tfrecordpath):        
        # needs to create the grid for each event, populate it, add to full tensor for event and save
        # per event
        # 0) drop unwanted columns from HL
        self.drop_variables_2_se(dataframe)
        # 1) create dictionaries for tfrecords
        # self.create_featuredesc2(dataframe)
        # path = tfrecordpath + "/dm%s_3in.tfrecords" % index
        length = 1 # self.calc_no_events(dataframe)
        # 2) convert df to numpy, reset indices on imvar_df
        dataframe = pd.DataFrame(dataframe, index=[0])
        # print(dataframe)
        npa = dataframe.to_numpy()
        # print(npa[0])
        imvar_dataframe.reset_index(drop=True, inplace=True)
        fulllen = imvar_dataframe.shape[0]
        del dataframe

        imvar_dataframe = (imvar_dataframe.groupby(["n_gammas_2"]\
            ).agg({'frac_energies': lambda x: x.tolist(),\
                'frac_momenta': lambda x: x.tolist(),'rot_phi':\
                    lambda x: x.tolist(),'rot_eta': lambda x: \
                        x.tolist()}).reset_index())
        print(imvar_dataframe)
        for a, row in imvar_dataframe.iterrows():
            print(a/fulllen)
            # feature_hl = tf.train.Feature(float_list=\
            #     tf.train.FloatList(value=npa[a].flatten()))
            feature_hl = npa[a].flatten()
            # function for creating grids with a dataframe row
            (grid1, grid2) = self.generate_grids(row, 21, 11)

            # feature_l = tf.train.Feature(int64_list=\
            #     tf.train.Int64List(value=grid1.flatten()))
            feature_l = grid1.flatten()
            # feature_s = tf.train.Feature(int64_list=\
            #     tf.train.Int64List(value=grid2.flatten()))
            feature_s = grid2.flatten()
            ex_hl =  tf.data.Dataset.from_tensor_slices(feature_hl, name="hl")
            ex_l =  tf.data.Dataset.from_tensor_slices(feature_l, name="large_image")
            ex_s =  tf.data.Dataset.from_tensor_slices(feature_s, name="small_image")
            ex_full = {"hl":ex_hl, "large_image":ex_l, "small_image":ex_s}# tf.data.Dataset.zip((ex_hl, ex_l, ex_s))
            # example = tf.data.Dataset.from_tensor_slices((feature_hl, feature_l, feature_s))#, name = ("hl", "large_image", "small_image"))

        # print(example)
        return ex_full

    def generate_datasets_anal_3(self, dataframe, imvar_dataframe, tfrecordpath):        
        self.drop_variables_2_se(dataframe)
        dataframe = pd.DataFrame(dataframe, index=[0])
        # print(dataframe)
        npa = dataframe.to_numpy()
        # print(npa[0])
        imvar_dataframe.reset_index(drop=True, inplace=True)
        fulllen = imvar_dataframe.shape[0]

        imvar_dataframe = (imvar_dataframe.groupby(["n_gammas_2"]\
            ).agg({'frac_energies': lambda x: x.tolist(),\
                'frac_momenta': lambda x: x.tolist(),'rot_phi':\
                    lambda x: x.tolist(),'rot_eta': lambda x: \
                        x.tolist()}).reset_index())
        print(imvar_dataframe)

        for a, row in imvar_dataframe.iterrows():
            test_inputs = []
            print("l_im_data")
            l_im_test = np.load(self.load_path + self.data_folder + "im_l_array_test.npy")[:self.test_length]
            test_inputs.append(l_im_test)

            print("s_im_data")
            s_im_test = np.load(self.load_path + self.data_folder + "im_s_array_test.npy")[:self.test_length]
            test_inputs.append(s_im_test)

            print("hl_data")
            X_test = pd.read_pickle(self.load_path + self.data_folder + "X_test_df.pkl").head(self.test_length)
            if self.drop_variables:
                vars_to_drop = ['pi2_E_2', 'pi3_E_2','n_gammas_2','sc1_Nclusters_2','tau_E_2',]
                X_test.drop(columns = vars_to_drop, inplace = True)
            test_inputs.append(X_test)

        return test_inputs

    def generate_datasets_anal_4(self, dataframe, imvar_dataframe, tfrecordpath):        
        # needs to create the grid for each event, populate it, add to full tensor for event and save
        # per event
        # 0) drop unwanted columns from HL
        self.drop_variables_2_se(dataframe)
        # 1) create dictionaries for tfrecords
        # self.create_featuredesc2(dataframe)
        path = tfrecordpath + "/dm.tfrecords"
        length = 1 # self.calc_no_events(dataframe)
        dataframe = pd.DataFrame(dataframe, index=[0])
        # 2) convert df to numpy, reset indices on imvar_df
        # print('Writing')
        npa = dataframe.to_numpy()
        imvar_dataframe.reset_index(drop=True, inplace=True)
        fulllen = imvar_dataframe.shape[0]
        del dataframe
        imvar_dataframe = (imvar_dataframe.groupby(["n_gammas_2"]\
            ).agg({'frac_energies': lambda x: x.tolist(),\
            'frac_momenta': lambda x: x.tolist(),'rot_phi':\
                lambda x: x.tolist(),'rot_eta': lambda x: \
                    x.tolist()}).reset_index())

        event_dict = {}
        for a, row in imvar_dataframe.iterrows():
            # print(a/fulllen)
            event_dict["hl"] = tf.train.Feature(float_list=\
                tf.train.FloatList(value=npa[a].flatten()))
            # print(event_dict)
            # function for creating grids with a dataframe row
            (grid1, grid2) = self.generate_grids(row, 21, 11)

            event_dict["large_image"] = tf.train.Feature(int64_list=\
                tf.train.Int64List(value=grid1.flatten()))
            event_dict["small_image"] = tf.train.Feature(int64_list=\
                tf.train.Int64List(value=grid2.flatten()))
        example = tf.train.Example(features=tf.train.Features(feature=event_dict))
            # print(example)
        return example.SerializeToString()

    def generate_dataframes_anal_multi(self, dataframe, imvar_dataframe, tfrecordpath):        
        # needs to create the grid for each event, populate it, add to full tensor for event and save
        # per event
        # 0) drop unwanted columns from HL
        self.drop_variables_2(dataframe)
        # 1) create dictionaries for tfrecords
        self.create_featuredesc2(dataframe)
        # path = tfrecordpath + "/dm%s_3in.tfrecords" % index
        length = self.calc_no_events(dataframe)
        # 2) convert df to numpy, reset indices on imvar_df
        print('Writing')
        npa = dataframe.to_numpy()
        imvar_dataframe.reset_index(drop=True, inplace=True)
        fulllen = imvar_dataframe.shape[0]
        del dataframe
        large_image = []
        small_image = []
        for index, row in imvar_dataframe.iterrows():
            (grid1, grid2) = self.generate_grids(row, 21, 11)
            large_image.append(grid1)
            small_image.append(grid2)
        large_image = np.array(large_image)
        small_image = np.array(small_image)
        # data_list = []
        # data_list.append(large_image)
        # data_list.append(small_image)
        # data_list.append(npa) 
        data_list = [large_image, small_image, npa]

        print("done creating data list")
        return data_list

    def modify_by_decay_mode(self):
        for a in range(len(self.df_dm)):
            #print(a)
            self.modify_dataframe(self.df_dm[a])
            imv_dm = self.create_imvar_dataframe(self.df_dm[a])
            self.save_dataframe(imv_dm, "/imvar_df_dm%s.pkl" % a)
            self.save_dataframe(self.df_dm[a], "/df_m_dm%s.pkl" % a)
            # New name to indicate modified
        self.clear_dataframe()

    def modify_single_df(self, filepath, name, index):
        df = self.load_dataframe(filepath, name)
        self.modify_dataframe(df)
        imv_dm = self.create_imvar_dataframe(df)
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
# jez = pipeline(rootpath_load, rootpath_save)
# index = int(sys.argv[1])
# # Takes the index from an argument
# # index = 0
# print(index)
# jez.load_hl_imvar(jez.save_path + jez.object_folder, df_mod_names[index], imvar_names[index])
# jez.generate_datasets2(jez.hl_df, jez.imvar_df, jez.save_path + "/E_TFRecords", index)
# def run_modifications(filepath):
#     for a in range(len(names)):
#         print(a)
#         jez.modify_single_df(filepath, names[a], a+3)
# run_modifications(jez.save_path + jez.object_folder)
# df = jez.load_dataframe(jez.save_path + jez.object_folder, "/df_dm0.pkl")
