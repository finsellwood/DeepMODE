import pandas as pd
import numpy as np
from tensorflow import keras, Tensor
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, History, ModelCheckpoint
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from math import ceil
import datetime
import pickle
import matplotlib.pyplot as plt
import json
from parameters import paramdict

### TO DO ###
# Create dataset conversion
# Add rest of the pipeline?

rootpath_load = "/vols/cms/fjo18/Masters2021"
rootpath_save = "/vols/cms/fjo18/Masters2021"
default_filepath = "/vols/cms/fjo18/Masters2021/D_Models/Models3_DM2_no_pi0/LSH_model_0.711_20220216_133722"
mez_filepath = "/vols/cms/fjo18/Masters2021/D_Models/Models3_DM2/LSH_model_0.715_20220205_195903"

class parameter_parser:
    def __init__(self, param_dict_filepath = None, param_dict = None):
        if param_dict_filepath is not None:
            paramfile = open(param_dict_filepath, "r")
            # Ensure filepath ends in .json
            self.parameter_dictionary = json.loads(paramfile.read())
            self.load_parameters()
        elif param_dict is not None:
            self.parameter_dictionary = param_dict
            self.load_parameters()
        else:
            raise Exception("Either a filepath or dictionary must be provided")
    def load_parameters(self):
        param_dict = self.parameter_dictionary
        self.batch_size = param_dict["batch_size"]
        self.dense_layers = param_dict["dense_layers"]
        self.conv_layers = param_dict["conv_layers"]
        self.inc_dropout = param_dict["inc_dropout"]
        self.dropout_rate = param_dict["dropout_rate"]
        self.use_inputs = param_dict["use_inputs"]
        self.learningrate = param_dict["learningrate"]
        self.no_epochs = param_dict["no_epochs"]
        self.stop_patience = param_dict["stop_patience"]
        self.use_res_blocks = param_dict["use_res_blocks"]
        self.drop_variables = param_dict["drop_variables"]
        self.flat_preprocess = param_dict["flat_preprocess"]
        self.HL_shape = param_dict["HL_shape"]
        self.im_l_shape = param_dict["im_l_shape"]
        self.im_s_shape = param_dict["im_s_shape"]
        self.no_modes = param_dict["no_modes"]
        self.data_folder = param_dict["data_folder"]
        self.model_folder = param_dict["model_folder"]
        self.save_model = param_dict["save_model"]
        self.small_dataset = param_dict["small_dataset"]
        self.small_dataset_size = param_dict["small_dataset_size"]

    def reload_parameters(self, param_dict_filepath):
        # Can reload parameters from a file - can be used in load_model (hep_model method)
        paramfile = open(param_dict_filepath, "r")
            # Ensure filepath ends in .json
        self.parameter_dictionary = json.loads(paramfile.read())
        self.load_parameters()

class hep_model(parameter_parser):
    def __init__(self, load_path, save_path, model_filepath = None, param_dict = None):
        self.load_path = load_path
        self.save_path = save_path
        #keras.Model.__init__(self, inputs, outputs)
        # to give the parameters of the keras.Model class
        # this doesnt work since you have to build the model first
        # so cant give it inputs/outputs yet
        if model_filepath is not None:
            param_dict_filepath = model_filepath + "_params.json"
            self.model_path = model_filepath
        else:
            param_dict_filepath = None

        parameter_parser.__init__(self, param_dict_filepath, param_dict)
        # All params are now posessed by the instance
        self.loaded_data = False
        self.model_built = False
        self.loaded_mva_data = False
        self.calculated_roc_values = False
        self.calculated_roc_values_mva = False
        self.model_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_accuracy = 0.0

        # Bool to check that data is loaded in before training
        # Parameters I'm unlikely to change
        self.poolingsize = (2,2)
        self.kernelsize = (3,3)
    
    def set_model_name(self, modelname):
        self.model_name = modelname
        
    def choose_model_path(self, path):
        # self.load_path = load_path
        # self.model_folder = model_folder
        # self.model_name = model_name
        self.model_path = path

    def update_model_path(self):
        self.model_path = self.load_path + self.model_folder + self.model_name

    def load_data(self):
        print("Loading data")
        # I am going to load the training data into this object.
        # This may be a bad idea
        self.y_train = pd.read_pickle(self.load_path + self.data_folder + "y_train_df.pkl")
        self.y_test = pd.read_pickle(self.load_path + self.data_folder + "y_test_df.pkl")
        self.train_length = self.y_train.shape[0]
        self.test_length = self.y_test.shape[0]
        if self.small_dataset:
            self.train_length = int(self.small_dataset_size*.8)
            self.test_length = int(self.small_dataset_size*.2)
            self.y_train = self.y_train[:self.train_length]
            self.y_test = self.y_test[:self.test_length]
            print(self.train_length)
            print(self.test_length)
        l_im_train = []
        l_im_test = []
        s_im_train = []
        s_im_test = []
        X_train = pd.DataFrame()
        X_test = pd.DataFrame()
        # These need to be here so that the later operations don't break when you only use some inputs
        self.train_inputs = []
        self.test_inputs = []
        if self.use_inputs[0]:
            print("l_im_data")
            l_im_train = np.load(self.load_path + self.data_folder + "im_l_array_train.npy")[:self.train_length]
            self.train_inputs.append(l_im_train)
            del l_im_train
            l_im_test = np.load(self.load_path + self.data_folder + "im_l_array_test.npy")[:self.test_length]
            self.test_inputs.append(l_im_test)
            del l_im_test
        if self.use_inputs[1]:
            print("s_im_data")
            s_im_train = np.load(self.load_path + self.data_folder + "im_s_array_train.npy")[:self.train_length]
            self.train_inputs.append(s_im_train)
            del s_im_train
            s_im_test = np.load(self.load_path + self.data_folder + "im_s_array_test.npy")[:self.test_length]
            self.test_inputs.append(s_im_test)
            del s_im_test
        if self.use_inputs[2]:
            print("hl_data")
            X_train = pd.read_pickle(self.load_path + self.data_folder + "X_train_df.pkl").head(self.train_length)
            X_test = pd.read_pickle(self.load_path + self.data_folder + "X_test_df.pkl").head(self.test_length)
            if self.drop_variables:
                vars_to_drop = ['pi2_E_2', 'pi3_E_2','n_gammas_2','sc1_Nclusters_2','tau_E_2',]
                X_train.drop(columns = vars_to_drop, inplace = True)
                X_test.drop(columns = vars_to_drop, inplace = True)
                self.HL_shape = (X_train.head(1).shape[1],)
            self.train_inputs.append(X_train)
            self.test_inputs.append(X_test)
            del X_train, X_test
        self.loaded_data = True

    def load_mva_data(self):
        self.mva_train = pd.read_pickle(self.load_path + self.data_folder + "mva_train.pkl").head(self.train_length)
        self.mva_test = pd.read_pickle(self.load_path + self.data_folder + "mva_test.pkl").head(self.test_length)
        self.loaded_mva_data = True

    def create_featuredesc(self):
        # Creates feature descriptions for tfrecord datasets
        self.featurenames_hl = list(self.test_inputs[2].columns)
        # npa = X_test.to_numpy()
        self.feature_description = {}
        self.fd_hl = {}
        self.fd_im_l = {}
        self.fd_im_s = {}
        for a in range(self.HL_shape[0]):
            self.feature_description[self.featurenames_hl[a]] = tf.io.FixedLenFeature([],tf.float32,default_value=0.0)
            self.fd_hl[self.featurenames_hl[a]] = tf.io.FixedLenFeature([],tf.float32,default_value=0.0)
        self.feature_description["large_image"] = tf.io.VarLenFeature(tf.int64)
        self.feature_description["small_image"] = tf.io.VarLenFeature(tf.int64)
        self.fd_im_l["large_image"] = tf.io.VarLenFeature(tf.int64)
        self.fd_im_s["small_image"] = tf.io.VarLenFeature(tf.int64)

    def write_datasets(self):
        self.tf_filepath_train = self.load_path + "/E_TFRecords/TFRecords3_DM2/train_events.tfrecords"
        self.tf_filepath_test = self.load_path + "/E_TFRecords/TFRecords3_DM2/test_events.tfrecords"
        print("Saving training data")
        with tf.io.TFRecordWriter(self.tf_filepath_train) as writer:
            npa = self.train_inputs[2].to_numpy()
            for a in range(self.train_length):
                event_dict = {}
                for b in range(npa.shape[1]):
                    event_dict[self.featurenames_hl[b]] = tf.train.Feature(float_list=tf.train.FloatList(value=[npa[a][b]]))
                event_dict["large_image"] = tf.train.Feature(int64_list=tf.train.Int64List(value=self.train_inputs[0][a].flatten()))
                event_dict["small_image"] = tf.train.Feature(int64_list=tf.train.Int64List(value=self.train_inputs[1][a].flatten()))
                example = tf.train.Example(features=tf.train.Features(feature=event_dict))
                writer.write(example.SerializeToString())

        print("Saving test data")
        with tf.io.TFRecordWriter(self.tf_filepath_test) as writer:
            npa = self.test_inputs[2].to_numpy()
            for a in range(self.test_length):
                event_dict = {}
                for b in range(npa.shape[1]):
                    event_dict[self.featurenames_hl[b]] = tf.train.Feature(float_list=tf.train.FloatList(value=[npa[a][b]]))
                event_dict["large_image"] = tf.train.Feature(int64_list=tf.train.Int64List(value=self.test_inputs[0][a].flatten()))
                event_dict["small_image"] = tf.train.Feature(int64_list=tf.train.Int64List(value=self.test_inputs[1][a].flatten()))
                example = tf.train.Example(features=tf.train.Features(feature=event_dict))
                writer.write(example.SerializeToString())


    ### Framework functions ###
    def make_input_layer(self, shape, name) -> Tensor:
        output_l = keras.Input(shape=shape, name=name)
        return output_l

    def relu_bn(self, inputs: Tensor) -> Tensor:
        relu = layers.ReLU()(inputs)
        bn = layers.BatchNormalization()(relu)
        return bn

    def add_dense_layer(self, input_layer: Tensor, no_nodes, dropout, dropout_rate) -> Tensor:
        x = layers.Dense(no_nodes)(input_layer)
        output_l = self.relu_bn(x)
        if dropout:
            output_l = layers.Dropout(dropout_rate)(output_l)
        return output_l

    def add_conv_layer(self, input_layer: Tensor, no_filters, kernel_size, dropout, dropout_rate, pooling) -> Tensor:
        output_l = layers.Conv2D(no_filters, kernel_size, padding="same")(input_layer)
        output_l = self.relu_bn(output_l)
        if pooling:
            output_l = layers.MaxPooling2D(pool_size=self.poolingsize)(output_l)
        if dropout:
            output_l = layers.Dropout(dropout_rate)(output_l)
        return output_l

    def concatenate_layers(self, hl_layer: Tensor, l_im_layer: Tensor, s_im_layer: Tensor) -> Tensor:
        l_layer = layers.Flatten()(l_im_layer)
        s_layer = layers.Flatten()(s_im_layer)
        all_input_layers = [l_layer, s_layer, hl_layer]
        model_to_concat = []
        for a in range(len(self.use_inputs)):
            if self.use_inputs[a]:
                model_to_concat.append(all_input_layers[a])
        if sum(self.use_inputs)==1:
            output_l = model_to_concat[0]
        else:
            output_l = layers.concatenate(model_to_concat)
        return output_l
    
    def add_output_layer(self, input_layer: Tensor) -> Tensor:
        output = layers.Dense(self.no_modes, name = "Outputs", activation = "softmax")(input_layer)
        return output

    def initialise_model(self, input_hl: Tensor, input_l: Tensor, input_s: Tensor, output_layer: Tensor):
        full_inputs = [input_l,input_s,input_hl]
        model_inputs = []
        for a in range(len(self.use_inputs)):
            if self.use_inputs[a]:
                model_inputs.append(full_inputs[a])
        self.model = keras.Model(inputs=model_inputs, outputs=output_layer)

    def build_model(self):
        print("Building model")
        # Initialises self.model as a keras.Model instance with desired structure
        # Compiles and summarises model
        image_input_l = self.make_input_layer(self.im_l_shape, "l_input")
        y_l = image_input_l
        image_input_s = self.make_input_layer(self.im_s_shape, "s_input")
        y_s = image_input_s
        input_hl = self.make_input_layer(self.HL_shape, "hl_input")
        y_hl = input_hl

        ### large image convolutional structure ###
        for a in range(self.conv_layers[0][0]):
            # NO POOLING
            if self.flat_preprocess:
                no_filters = 32
            else:
                no_filters = 32 * (a+1)
            y_l = self.add_conv_layer(y_l, no_filters,self.kernelsize, self.inc_dropout, self.dropout_rate[0], pooling = False)
        
        for a in range(self.conv_layers[0][1]):
            y_l = self.add_conv_layer(y_l, 32 * (a+1), self.kernelsize, self.inc_dropout, self.dropout_rate[0], pooling = True)
        
        ### small image convolutional structure ###
        for a in range(self.conv_layers[1][0]):
            # NO POOLING
            if self.flat_preprocess:
                no_filters = 32
            else:
                no_filters = 32 * (a+1)
            y_s = self.add_conv_layer(y_s, no_filters,self.kernelsize, self.inc_dropout, self.dropout_rate[0], pooling = False)
        
        for a in range(self.conv_layers[1][1]):
            y_s = self.add_conv_layer(y_s, 32 * (a+1), self.kernelsize, self.inc_dropout, self.dropout_rate[0], pooling = True)
        
        ### high level dense structure ###
        for a in range(self.dense_layers[0][0]):
            if self.dense_layers[0][2]:
                y_hl = self.add_dense_layer(y_hl, ceil(self.dense_layers[0][1] * 0.5 ** a), self.inc_dropout, self.dropout_rate[1])
            else:
                y_hl = self.add_dense_layer(y_hl, self.dense_layers[0][1], self.inc_dropout, self.dropout_rate[1])
        
        ### final dense structure ###
        y = self.concatenate_layers(y_hl, y_l, y_s)
        for a in range(self.dense_layers[1][0]):
            if self.dense_layers[1][2]:
                y = self.add_dense_layer(y, ceil(self.dense_layers[1][1] * 0.5 ** a), self.inc_dropout, self.dropout_rate[1])
            else:
                y = self.add_dense_layer(y, self.dense_layers[1][1], self.inc_dropout, self.dropout_rate[1])
        
        output = self.add_output_layer(y)

        self.initialise_model(input_hl, image_input_l, image_input_s, output)
        self.model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=self.learningrate), metrics=["accuracy"],)
        self.model.summary()
        self.model_built = True   

    def load_model(self):
        # Parameter file is a boolean
        print("Loading model")
        # self.choose_model_path(filepath)
        # if parameter_file:
        #     self.reload_parameters(self.model_path + "_params.json")
        self.model = keras.models.load_model(self.model_path)
        self.model_built = True

    def save_model_parameters(self, update_name:bool):
        if update_name:
            input_string = ''
            inputflags = ['L', 'S', 'H']
            for a in range(len(self.use_inputs)):
                if self.use_inputs[a]:
                    input_string += inputflags[a]
            self.model_name = "%s_model_%.3f_%s" % (input_string, self.model_accuracy, self.model_datetime)
            self.model_path = self.save_path + self.model_folder + self.model_name
        param_file = open(self.model_path + "_params.json", 'w')
        json.dump(self.parameter_dictionary, param_file)
        param_file.close()

    def train_model(self):
        
        if self.loaded_data == False:
            raise Exception("Data has not been loaded yet")
        if self.model_built == False:
            raise Exception("Model has not been built yet")

        print("Training model")

        self.save_model_parameters(True)
        # So that the model can be recovered if killed

        self.early_stop = EarlyStopping(monitor = 'val_loss', patience = self.stop_patience)
        self.history = History()
        self.checkpoint_filepath = self.save_path + "/Checkpoints/checkpoint"
        self.model_checkpoint = ModelCheckpoint(filepath = self.checkpoint_filepath, monitor = "val_loss", mode = "min",\
        verbose = 0, save_best_only = True, save_weights_only = True)

        print(self.parameter_dictionary)
        self.model.fit(self.train_inputs, self.y_train, 
                    batch_size = self.batch_size, 
                    epochs = self.no_epochs, 
                    callbacks = [self.history, self.early_stop, self.model_checkpoint], 
                    validation_data = (self.test_inputs, self.y_test))
        self.model.load_weights(self.checkpoint_filepath)
        print("Completed training")

    def predict_results(self):
        self.prediction = self.model.predict(self.test_inputs)
        idx = self.prediction.argmax(axis=1)
        self.y_pred = (idx[:,None] == np.arange(self.prediction.shape[1])).astype(float)

    def predict_results_mva(self):
        # Rearranges the MVA dataframe based on the predicted mode
        # For ROC curve generation
        if self.loaded_mva_data == False:
            raise Exception("MVA data not loaded")
        mva_modes = ["t_dm0raw_2", "t_dm1raw_2", "t_dm2raw_2", \
                    "t_dm10raw_2", "t_dm11raw_2", "t_dmotherraw_2",]
        self.prediction_mva = pd.DataFrame(self.mva_test[mva_modes]).to_numpy()
        self.no_mva_modes = self.no_modes

    def analyse_model(self):
        self.predict_results()
        self.model_accuracy = accuracy_score(self.y_test, self.y_pred)
        print(self.model_accuracy)

    def model_save(self, update_name: bool):
        # If loaded model from a file, don't want to rename or save in different place
        print("Saving model")
        self.save_model_parameters(update_name)
        self.model.save(self.model_path)

        # Saves model parameters in a corresponding .txt file
        with open(self.model_path + "_history", 'wb') as file_pi:
            pickle.dump(self.history.history, file_pi)
 
    def load_from_checkpoint(self):
        if self.model_built == False:
            raise Exception("Need a blank model to load checkpoints onto")
        self.checkpoint_filepath = self.save_path + "/Checkpoints/checkpoint"
        self.model.load_weights(self.checkpoint_filepath)
    
    def unreaper(self):
        # Model must be initialised with the original parameters
        self.load_from_checkpoint()
        self.train_model()
        self.model_save(True)
        # Save model with a new name    
    def unreaper_ver2(self):
        self.load_from_checkpoint()
        self.analyse_model()
        self.model_save(True)
   
    def plot_timeline(self):
        self.history = pickle.load(open(self.model_path + '_history',  'rb'))
        epochs = range(1, len(self.history["loss"])+1)
        fig, ax = plt.subplots(2,1)
        # Extract loss on training and validation ddataset and plot them together
        ax[0].plot(epochs, self.history["loss"], "o-", label="Training")
        ax[0].plot(epochs, self.history["val_loss"], "o-", label="Test")
        ax[0].set_xlabel("Epochs"), ax[0].set_ylabel("Loss")
        ax[0].set_yscale("log")
        ax[0].legend()

        # do the same for the accuracy:
        # Extract number of run epochs from the training history
        epochs2 = range(1, len(self.history["accuracy"])+1)

        # Extract accuracy on training and validation ddataset and plot them together
        ax[1].plot(epochs2, self.history["accuracy"], "o-", label="Training")
        ax[1].plot(epochs2, self.history["val_accuracy"], "o-", label="Test")
        ax[1].set_xlabel("Epochs"), ax[1].set_ylabel("accuracy")
        ax[1].legend()
        
        plt.savefig(self.model_path + '_tl' + '.png', dpi = 100)

    def calc_eff_purity(self):
        flatpred = np.argmax(self.y_pred, axis=-1)
        flattest = np.argmax(self.y_test, axis=-1)
        truelabelefficiency = confusion_matrix(flattest, flatpred, normalize = 'true')
        truelabelpurity = confusion_matrix(flattest, flatpred, normalize = 'pred')
        self.eff = [truelabelefficiency[a,a] for a in range(self.no_modes)]       
        self.pur = [1 - truelabelpurity[a,a] for a in range(self.no_modes)]

    def plot_confusion_matrices(self):
        if self.model_accuracy == 0.0:
            raise Exception("Model has not made any predictions yet")
        flatpred = np.argmax(self.y_pred, axis=-1)
        flattest = np.argmax(self.y_test, axis=-1)
        truelabelefficiency = confusion_matrix(flattest, flatpred, normalize = 'true')
        truelabelpurity = confusion_matrix(flattest, flatpred, normalize = 'pred')

        plt.rcParams.update({'figure.autolayout': True})
        labellist = [r'$\pi^{\pm}$', r'$\pi^{\pm} \pi^0$', r'$\pi^{\pm} 2\pi^0$', r'$3\pi^{\pm}$', r'$3\pi^{\pm} \pi^0$', 'other']
        fig, ax = plt.subplots(1,2)
        plt.tight_layout()
        fig.set_size_inches(12, 8)

        ax[0].imshow(truelabelefficiency, cmap = 'Blues')
        for i in range(truelabelefficiency.shape[0]):
            for j in range(truelabelefficiency.shape[1]):
                if truelabelefficiency[i, j] > 0.5:
                    text = ax[0].text(j, i, round(truelabelefficiency[i, j], 3),
                                ha="center", va="center", color="w")
                else:
                    text = ax[0].text(j, i, round(truelabelefficiency[i, j], 3),
                                ha="center", va="center", color="black")

                
        ax[0].set_title('Efficiency')
        labellist = labellist[:self.no_modes]
        ticklocs = np.linspace(0, len(labellist)-1, len(labellist))    
        ax[0].set_xticks(ticklocs)
        ax[0].set_yticks(ticklocs)
        ax[0].set_xticklabels(labellist)
        ax[0].set_yticklabels(labellist)
        ax[0].set_xlabel('Predicted Mode')
        ax[0].set_ylabel('True Mode')


        ax[1].imshow(truelabelpurity, cmap = 'Blues')
        for i in range(truelabelefficiency.shape[0]):
            for j in range(truelabelefficiency.shape[1]):
                if truelabelpurity[i, j] > 0.5:
                    text = ax[1].text(j, i, round(truelabelpurity[i, j], 3),
                                ha="center", va="center", color="w")
                else:
                    text = ax[1].text(j, i, round(truelabelpurity[i, j], 3),
                                ha="center", va="center", color="black")

        ax[1].set_title('Purity')
        ax[1].set_xticks(ticklocs)
        ax[1].set_yticks(ticklocs)
        ax[1].set_xticklabels(labellist)
        ax[1].set_yticklabels(labellist)
        ax[1].set_xlabel('Predicted Mode')
        ax[1].set_ylabel('True Mode')


        plt.savefig(self.model_path + '_cm_' + '.png', dpi = 100)

    def calc_roc_values(self, roc_mva = False):
        if roc_mva == False:
            self.fpr = dict()
            self.tpr = dict()
            self.roc_auc = dict()
            for i in range(self.no_modes):
                self.fpr[i], self.tpr[i], _ = roc_curve(self.y_test[:,i], self.prediction[:,i])
                self.roc_auc[i] = auc(self.fpr[i], self.tpr[i])
            self.calculated_roc_values = True
        
        else:
            self.fpr_mva = dict()
            self.tpr_mva = dict()
            self.roc_auc_mva = dict()
            for i in range(self.no_mva_modes):
                self.fpr_mva[i], self.tpr_mva[i], _ = roc_curve(self.y_test[:,i], self.prediction_mva[:,i])
                self.roc_auc_mva[i] = auc(self.fpr_mva[i], self.tpr_mva[i])
            self.calculated_roc_values_mva = True

    def calc_wrong_roc_values(self, roc_mva = False):
        if roc_mva == False:
            self.fpr = dict()
            self.tpr = dict()
            self.roc_auc = dict()
            for i in range(self.no_modes):
                self.fpr[i], self.tpr[i], _ = roc_curve(self.y_test[:,i], self.y_pred[:,i])
                self.roc_auc[i] = auc(self.fpr[i], self.tpr[i])
            self.calculated_roc_values = True
        
        else:
            self.fpr_mva = dict()
            self.tpr_mva = dict()
            self.roc_auc_mva = dict()
            for i in range(self.no_mva_modes):
                self.fpr_mva[i], self.tpr_mva[i], _ = roc_curve(self.y_test[:,i], self.prediction_mva[:,i])
                self.roc_auc_mva[i] = auc(self.fpr_mva[i], self.tpr_mva[i])
            self.calculated_roc_values_mva = True


    def plot_two_roc_curves(self, fpr1, tpr1, roc_auc1, fpr2, tpr2, roc_auc2, one_roc_graph = False, plot_effpur = False):
        if one_roc_graph:
            fig2, ax2 = plt.subplots(1, 1)

            for i in range(self.no_modes):
                ax2.plot(fpr1[i], tpr1[i], label="Model 1 mode %s ROC curve (area = %0.2f)" % (i, roc_auc1[i]))
                ax2.plot(fpr2[i], tpr2[i], label="Model 2 mode %s ROC curve (area = %0.2f)" % (i, roc_auc2[i]))
                ax2.set_xlim([0,1])
                ax2.set_ylim([0,1.05])
                ax2.set_xlabel("False Positive Rate")
                ax2.set_ylabel("True Positive Rate")
                ax2.set_title("ROC_Curve for One Prong Tau Decays")
                ax2.legend()
                plt.savefig(self.model_path + '_roc_' + '.png', dpi = 100)
        else:
            print('plotting')
            fig2, ax2 = plt.subplots(1,self.no_modes)
            fig2.set_size_inches(12,4)

            for i in range(self.no_modes):
                ax2[i].plot(fpr1[i], tpr1[i], label="Model 1 ROC Curve (area = %0.2f)" % roc_auc1[i])
                if plot_effpur == True:
                    ax2[i].plot(self.pur[i], self.eff[i], 'x')
                ax2[i].plot(fpr2[i], tpr2[i], label="Model 2 ROC Curve (area = %0.2f)" % roc_auc2[i])
                ax2[i].set_xlim([0,1])
                ax2[i].set_ylim([0,1.05])
                ax2[i].set_xlabel("False Positive Rate")
                ax2[i].set_ylabel("True Positive Rate")
                ax2[i].set_title("ROC_Curve for Mode %s" % i)
                for axes in ax2.flat:
                    axes.label_outer()
                ax2[i].legend()
                fig2.tight_layout()
                ax2[i].set_aspect(1)
            plt.savefig(self.model_path + '_roc_multigraph' + '.png', dpi = 500)

    def plot_roc_curves_vs_mva(self, one_roc_graph = False):
        if self.loaded_mva_data == False:
            raise Exception("MVA data not loaded")
        if self.model_accuracy == 0.0:
            raise Exception("Model has not made any predictions yet")
        self.predict_results_mva()

        self.calc_wrong_roc_values(roc_mva=False)
        self.calc_wrong_roc_values(roc_mva=True)

        self.plot_two_roc_curves(self.fpr, self.tpr, self.roc_auc, self.fpr_mva, self.tpr_mva, self.roc_auc_mva, one_roc_graph, True)

    def compare_roc_curves(self, other):
        if self.calculated_roc_values == False:
            self.calc_roc_values(False)
        if other.calculated_roc_values == False:
            other.calc_roc_values(False)
        self.plot_two_roc_curves(self.fpr, self.tpr, self.roc_auc, other.fpr, other.tpr, other.roc_auc, False)
    ### META COMMANDS ###

    def do_your_thing(self):
        self.load_data()
        self.build_model()
        self.train_model()
        self.analyse_model()
        self.model_save(update_name = True)

    def retrain_model(self):
        self.load_data

    def prep_for_analysis(self, small_dataset = False):
        # From initialising object, prepares to produce roc curves etc
        self.small_dataset = small_dataset
        self.load_data()
        self.load_mva_data()
        self.load_model()
        self.analyse_model()
# jez = hep_model(paramdict, rootpath_load, rootpath_save)
# jez.do_your_thing()

# jez = hep_model(rootpath_load, rootpath_save, model_filepath = rootpath_load + "/D_Models/Models3_DM2_no_pi0/LSH_model_0.000_20220216_131708")
# jez.no_epochs = 14
# jez.build_model()
# jez.load_data()
# jez.unreaper()

