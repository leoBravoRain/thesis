#!/usr/bin/env python
# coding: utf-8

# # Note
# This notebook is to train the encoder as a classifier with the idea of validate the encoder architecture first and then use this to train the VAE.

# # Parameters to experiment

# In[1]:


# this must to be same as the own model
sizePercents = [0.1, 0.3, 0.5, 0.7, 1]


# trainingOnGuanaco = False
trainingOnGuanaco = True


# In[2]:


# training on guanaco
# ATENTION: if it is going to run on guanaco:
# 1) comment the %matplotlib magic in next block and any magic (something like %code)
# 2) Change to True the trainingOnGuanaco vairbale
# 3) set epoch with an appropiate number
# 4) add comment to experiemnts
# 5) Add this file as python file 
# 6) Change launchJobOnGuanaco file to run this file but with python format
# trainingOnGuanaco = False

# seed to generate same datasets
seed = 0


# In[3]:


# libraries for RF
import os
os.environ["MKL_NUM_THREADS"]="1"
# print(os.environ["MKL_NUM_THREADS"])
import sys
from os.path import join, exists
import pandas as pd
from joblib import Parallel, delayed, dump
import pickle
from itertools import zip_longest 
import turbofats
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# # features + RF

# In[4]:


# get data


# In[5]:


from joblib import load, dump
# # load file
rf = load('../experiments/comparingModels/seed' + str(seed) + '/RF/trainedRF.joblib')


# ## compute features

# In[6]:


# %%timeit -n 1 -r 1

plasticc_path = "/home/shared/astro/PLAsTiCC/" if trainingOnGuanaco else "/home/leo/Downloads/plasticData/"
#plasticc_path = "/home/shared/astro/PLAsTiCC/"
# plasticc_path = "/home/leo/Downloads/plasticData/"

# print("RF")
def compute_fats_features(batch_names):
    """
    Receives a list of file names and a path

    Returns a dataframe with the features
    """
    # TODO: STUDY BIASED FEATURES
    feature_list = ['CAR_sigma','CAR_mean', 'Meanvariance', 'Mean',                          'PercentAmplitude', 'Skew', 'AndersonDarling', 'Std', 'Rcs', 'StetsonK',
                         'MedianAbsDev', 'Q31', 'Amplitude', 'PeriodLS_v2', 'Harmonics',
                'Autocor_length', 'SlottedA_length', 'StetsonK_AC',  'Con', 'Beyond1Std',
                'SmallKurtosis', 'MaxSlope','MedianBRP', 'PairSlopeTrend', 
                'LinearTrend', 'Eta_e', 'Period_fit_v2', 'PeriodPowerRate',
                'Psi_CS_v2', 'Psi_eta_v2', 'StructureFunction_index_21', 'Pvar', 'StructureFunction_index_31',
                'ExcessVar', 'IAR_phi']
#     features = []
#     print(len(feature_list))
    df_features = pd.DataFrame()
    
#     print(df_features.shape)
    
#     print(batch_names)
    for name in batch_names:
#         print(name)
        # Check that name is valid
        if name is None:
            continue
        # Read light curve
        lc_data = parse_light_curve_data(name)
#         print(lc_data)
        if lc_data is None:
            continue
            
        # shape: [6 (channels) x 48 (features) = 288]
        features_lc = []
        
        # each filter or channel
        for fid in range(6):
            
            # Check that lc has more than 10 points
            #if lc_data.shape[0] < 7:
            #    print("Light curve %s has less than 7 points, skipping" %(name))
            #    continue
            
            
            # Compute features
            
            # filter by channel
            df_lc_fid = lc_data.loc[lc_data.fid == fid][["mjd", "mag", "err"]]
            
            # compute feature
            feature_space = turbofats.FeatureSpace(feature_list=feature_list,
                                                      data_column_names=["mag", "mjd", "err"])
            features_fid = feature_space.calculate_features(df_lc_fid)
            
            # rename column on data
            # shape: [48]
            features_fid = features_fid.rename(lambda x: x+"_"+str(fid), axis='columns')
            
#             print(features_fid.shape)
            
            features_lc.append(features_fid)
            
#         pd.concat()

        # concat all features of all channel of light curve
        dfOneLC = pd.concat(features_lc, axis=1)
#         print(dfOneLC.shape)
        
        # concat the light curve to full dataset
        df_features = pd.concat([df_features, dfOneLC])
#         print(df_features.shape)
        
#     # concat all data
#     print(df_features.shape)
    
    
    # do inference
    predictions = rf.predict(df_features.fillna(-1000).values)
    
#     print(predictions.shape)
    
def split_list_in_chunks(iterable, chunk_size, fillvalue=None):
    """
    Receives an iterable object (i.e. list) and a chunk size

    Returns an iterable object with the same elements on of the original but arranged in chunks
    """
    args = [iter(iterable)] * chunk_size
    return zip_longest(*args, fillvalue=fillvalue)

def parse_light_curve_data(light_curve_id):
    path_to_light_curve = join(plasticc_path, "light_curves", str(int(light_curve_id))+".pt")
    if not exists(path_to_light_curve):
        if raise_error:
            raise FileNotFoundError("File not found at: %s" %(path_to_light_curve))
        return None
    with open(path_to_light_curve, "rb") as f:
        lc_torch = torch.load(f)
    data = []
    for band in range(6):
        mask = lc_torch[band, -1, :] == 1
        tmp = lc_torch[band, :3, mask].T
        data.append(np.concatenate((tmp.numpy(), np.ones(shape=(tmp.shape[0], 1))*band),axis=1))
    df_lc = pd.DataFrame(data=np.concatenate(data, axis=0), columns=['mjd', 'mag', 'err', 'fid'])
    df_lc.index = [int(light_curve_id)]*len(df_lc.index.unique())
    return df_lc


# if __name__ == "__main__":

#     path = "/home/lbravo/" if trainingOnGuanaco else "/home/leo/Desktop"
    
#     #with open("/home/phuijse/plasticc_rf/dataset_ids_before_balancing.pkl", "rb") as f:
# #     with open(path + "/thesis/work/thesis/experiments/" + str(number_experiment) +"/seed" + str(seed) + "/maxClass15k/dataset_ids_before_balancing.pkl", "rb") as f:
#     with open(path + "/thesis/work/thesis/experiments/99/seed" + str(seed) + "/maxClass15k/dataset_ids_before_balancing.pkl", "rb") as f:
    
#         ids = pickle.load(f)
        
#         Parallel(n_jobs=1)(delayed(compute_fats_features)(batch_names) for batch_names in split_list_in_chunks(ids["train"], 100))


# In[7]:


path = "/home/lbravo/" if trainingOnGuanaco else "/home/leo/Desktop"
    
for sizePercent in sizePercents:
    
    with open(path + "/thesis/work/thesis/experiments/99/seed" + str(seed) + "/maxClass15k/dataset_ids_before_balancing_" + str(sizePercent) + ".pkl", "rb") as f:
    
        ids = pickle.load(f)
        print(ids["train"].shape[0])
        
        get_ipython().run_line_magic('timeit', '-n 1 -r 1 Parallel(n_jobs=1)(delayed(compute_fats_features)(batch_names) for batch_names in split_list_in_chunks(ids["train"], 100))')

