import specific_functions as SF
import numpy as np
from joblib import dump, load    


def generate_model(model_features, files_for_features, specific_info_for_data, timelags_for_features, outputs, hyperparameters, save_model):
   
    print("reading training data")
    training_data, training_labels = SF.load_data(model_features=model_features, files_for_features=files_for_features, specific_info_for_data=specific_info_for_data, timelags_for_features=timelags_for_features, outputs_for_labels=outputs_for_labels)

    if specific_info_for_data["PCA switch"]:     
        print("running PCA")   
        training_data, features = SF.format_features_to_frame(data_in=training_data, model_features=model_features, timelags_for_features=timelags_for_features)
        pca_model = SF.build_model_PCA(training_data, specific_info_for_data=specific_info_for_data)
        training_data = SF.apply_PCA_model(pca_model, features, training_data, PCA_dimensionality=specific_info_for_data["PCA dimensionality of features"])
   
    print("defining NN model")   
    
    NN_model_def = SF.NN_model_def(training_data.shape[1], training_labels.shape[1], hidlayer_1=int(hyperparameters["hidden layers scaling"]*training_data.shape[1]), hidlayer_2=int(hyperparameters["hidden layers scaling"]*training_data.shape[1]), hidlayer_3=(hyperparameters["hidden layers scaling"]*training_data.shape[1]), dropout_value=hyperparameters["model_dropout"])
           
    print("training NN model")
    NN_model = SF.NN_model_train(training_data, training_labels, model_input=NN_model_def, epochs_input=hyperparameters["number_of_epochs"], batch_input=hyperparameters["batch_value"], splits_input=hyperparameters["validation_split"])

    print("evaluating NN model")
    NNe.NN_model_evaluate(train_data, train_labels, NN_model)

    if save_model["switch"]:
        print("saving model")
        NN_model.save(save_model["path_to_NN_model"])

    return NN_model



def predict_data(model_features, files_for_features, specific_info_for_data, timelags_for_features, outputs_for_labels, model):

    print("reading data")
    data, labels = SF.load_data(model_features=model_features, files_for_features=files_for_features, specific_info_for_data=specific_info_for_data, timelags_for_features=timelags_for_features, outputs_for_labels=outputs_for_labels)

    if specific_info_for_data["PCA switch"]:     
        print("applying PCA")
        data, features = SF.format_features_to_frame(data_in=data, model_features=model_features, timelags_for_features=timelags_for_features)
        pca_model = load(specific_info_for_data["PCA model file"]) 
        data = SF.apply_PCA_model(pca_model, features, data, PCA_dimensionality=specific_info_for_data["PCA dimensionality of features"])    

    print("producing predictions")
    predictions = NNe.predict(model, data)
       
    return predictions, labels
 
