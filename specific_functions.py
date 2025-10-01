import tensorflow as tf
import os
from skimage import io, transform
import numpy as np
from skimage.color import rgb2gray
from netCDF4 import Dataset
import pandas as pd
from sklearn.decomposition import PCA
from joblib import dump, load
#from copy import deepcopy as cp
#from tf.keras.wrappers.scikit_learn import KerasRegressor
#from matplotlib import pyplot as plt



    
def load_data(model_features, files_for_features, specific_info_for_data, timelags_for_features, outputs):

    n_vars = 0
    max_timelag = 0

    for type_feature in list(model_features):

        n_vars += len(model_features[type_feature])*len(timelags_for_features[type_feature])
        max_timelag = max(max_timelag, np.amax(timelags_for_features[type_of_feature]))

        
# reading nc files
         
    daysmin = int(max(0, specific_info_for_data["period"][0]-max_timelag))
    daysmax = int(specific_info_for_data["period"][1] - max_timelag)

# the length of the selected data period
    
    n_days = daysmax - daysmin    
    feature_data = np.zeros((n_days, n_vars))
  
    item = 0  

# read data, reshuffle them randomly and pick the sub-set with the desired size

    for type_feature in list(model_features):

        input_file = Dataset(files_for_features[type_feature])

        for feature in model_features[type_feature]:
            for timelag in timelags_for_features[type_feature]:

                if (feature != "ChlTot_an") | ("ChlTot_an" in input_file.variables.keys()): 
                    feature_data[:,item] = input_file.variables[feature][daysmin + max_timelag - timelag : daysmax + max_timelag - timelag].filled()
                else: 
                    feature_data[:,item] = input_file.variables["P1_Chl_an"][daysmin + max_timelag - timelag : daysmax + max_timelag - timelag].filled() 
                    feature_data[:,item] += input_file.variables["P2_Chl_an"][daysmin + max_timelag - timelag : daysmax + max_timelag - timelag].filled()
                    feature_data[:,item] += input_file.variables["P3_Chl_an"][daysmin + max_timelag - timelag : daysmax + max_timelag - timelag].filled()
                    feature_data[:,item] += input_file.variables["P4_Chl_an"][daysmin + max_timelag - timelag : daysmax + max_timelag - timelag].filled()   
                item+=1 

        input_file.close()
    
        
    file_for_labels = Dataset(outputs["file"])
    labels = file_for_labels.variables[outputs["variable"]][daysmin+max_timelag:daysmax+max_timelag,outputs["layers"]].filled()  
            
    return feature_data, labels

def format_features_to_frame(data_in, model_features, timelags_for_features):

    features = []
    for type_feature in list(model_features):
        for feature in model_features[type_feature]:
            for timelag in timelags_for_features[type_feature]:
                features.append(feature+"_"+str(timelag))

    data_in_frame = pd.DataFrame(data_in, columns = features)
 
    return data_in_frame, features


def build_model_PCA(data_in_frame, specific_info_for_data):
    
    pca = PCA(n_components=specific_info_for_data["PCA dimensionality of features"])

    if specific_info_for_data["save PCA model switch"]:
        pca.fit(data_inter)
        dump(pca, specific_info_for_data["PCA model file"])    

    return pca

def apply_PCA_model(PCA_input, features, data_in_frame, PCA_dimensionality):

    principalComponents = pca.fit_transform(data_in_frame)

    column = []    
    for i in range(0,PCA_dimensionality):    
        column.append("principal_component_"+str(i+1))    
        
    principalDf = pd.DataFrame(data = principalComponents, columns = column)    
    data_out = principalDf.to_numpy()

    return data_out    
    

def NN_model_def(data_input_size, data_output_size, hidlayer_1, hidlayer_2, hidlayer_3, dropout_value):    
    model_out = tf.keras.Sequential([tf.keras.layers.Dense(hidlayer_1, input_dim=data_input_size, kernel_initializer='normal', activation='relu'), tf.keras.layers.Dropout(dropout_value), tf.keras.layers.Dense(hidlayer_2, kernel_initializer='normal', activation='relu'), tf.keras.layers.Dropout(dropout_value), tf.keras.layers.Dense(hidlayer_3, kernel_initializer='normal', activation='relu'), tf.keras.layers.Dropout(dropout_value), tf.keras.layers.Dense(data_output_size, kernel_initializer='normal', activation='linear')])
    model_out.compile(loss='mean_squared_error', optimizer='adam')
    return model_out    
    
def NN_model_train(data_input, labels_input, model_input, epochs_input, batch_input, splits_input):
    model_input.fit(data_input, labels_input, epochs=epochs_input, batch_size=batch_input, validation_split=splits_input)
    return model_input

def NN_model_evaluate(data_input, labels_input, model_input):    
    accuracy = model_input.evaluate(data_input, labels_input)
    print('Accuracy: %.2f' % (accuracy*100))    
   
def predict(model_input, data_input):
    predictions_out = model_input.predict(data_input)
    return predictions_out
    

