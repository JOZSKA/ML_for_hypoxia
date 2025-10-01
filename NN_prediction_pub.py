
# Jozef Skakala, PML, 21/03/2022
import tensorflow as tf
import basic_tasks as BT


class O2_NN_prediction_in_1D:

    def __init__(self, generate_new, **kwargs):
    
        self.description = "Prediction of oxygen at different layers from observable data at the L4 location."
        self.author = "Jozef Skakala"


        if "model_features" in kwargs:       # "model_features" lists the variables to be included in the inputs. The features are clustered in different sets that each correspond to the same input netCDF file and the same range of timelags.   
            model_features = kwargs["model_features"]
            files_for_features = kwargs["files_for_features"]  # each set of features has an associated source netCDF file
            specific_info_for_data = kwargs["specific_info_for_data"]   # some necessary info, e.g what is the time interval for the (training/test) data, whether PCA will be used, what number of features will be obtained after the PCA
            timelags_for_features = kwargs["timelags_for_features"]  # for each set member these are timelags to be used
        else:     # if the above parameters weren't supplied as arguments these are their default values...
            path = "/home/jos/Documents/REPLACE/data/"
            model_features = {"model_inputs":["T_an", "ChlTot_an"], "riverine_inputs":["p_an"]}
            files_for_features = {"model_inputs":path+"L4_model_outputs.nc", "riverine_inputs":path+"L4_riverine_deposition.nc"}
            specific_info_for_data = kwargs["period":[0,int(365.25*20)], "PCA switch":True, "PCA dimensionality of features":20, "save PCA model switch":True, "PCA model file": path+"../models/pca_model.joblib"]
            timelags_for_features = {"model_inputs":np.arange(0,41), "riverine_inputs":np.arange(0,41)}

        if "outputs" in kwarg:
            outputs_for_labels = kwarg["outputs_for_labels"]   # what are we predicting, at which vertical layers and where the label data can be found
        else:
            outputs_for_labels = {"variable":"O2_an", "layers":np.arange(0,50), "file":path+"L4_model_outputs.nc"}

        if "hidden_layers_scaling" in kwargs:   # hyperparameters that define the structure of the NN model
            hyperparameters = {"hidden_layers_scaling": kwargs["hidden_layers_scaling"]}   # this is the linear proportionality constant between the number of neurons in each hidden layer and the number of features
        else: 
            hyperparameters = {"hidden_layers_scaling": 10.0}

        if "model_dropout" in kwargs:
            hyperparameters["model_dropout"] = kwargs["model_dropout"]  #the dropout = aproportion of neurons randomly switched off to prevent overfitting 
        else: 
            hyperparameters["model_dropout"] = 0.3

        if "number_of_epochs" in kwargs:
            hyperparameters["number_of_epochs"] = kwargs["number_of_epochs"]  # number of epochs
        else: 
            hyperparameters["number_of_epochs"] = 15

        if "validation_split" in kwargs:
            hyperparameters["validation_split"] = kwargs["validation_split"]  # ratio of validation to total (training+validation) data
        else: 
            hyperparameters["validation_split"] = 0.2

        if "batch_value" in kwargs:
            hyperparameters["batch_value"] = kwargs["batch_value"]
        else: 
            hyperparameters["batch_value"] = 60

        if "save_model" in kwargs:      # here we determine whether the model will be stored somewhere
            save_model = kwargs["save_model"]
        else: 
            save_model = {"switch":False, "path_to_NN_model":path+"../models/final_model"}
     
 

        if generate_new:
            self.generated_model = BT.generate_model(model_features=model_features, files_for_features=files_for_features, specific_info_for_data=specific_info_for_data, timelags_for_features=timelags_for_features, outputs_for_labels=outputs_for_labels, hyperparameters=hyperparameters, save_model=save_model)
            (self.prediction, self.labels) = BT.predict_data(model_features=model_features, files_for_features=files_for_features, specific_info_for_data=specific_info_for_data, timelags_for_features=timelags_for_features, outputs_for_labels=outputs_for_labels, model=self.generated_model)

        else:
            self.loaded_model = tf.keras.models.load_model(kwargs["path_to_existing_model"])
            (self.prediction, self.labels) = BT.predict_data(model_features=model_features, files_for_features=files_for_features, specific_info_for_data=specific_info_for_data, timelags_for_features=timelags_for_features, outputs_for_labels=outputs_for_labels, model=self.loaded)


    def model(self):
        if generate_new:
            return self.generated_model
        else:
            return self.loaded_model

    def prediction(self):
        return self.prediction, self.labels


