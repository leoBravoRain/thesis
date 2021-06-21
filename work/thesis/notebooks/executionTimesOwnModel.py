#!/usr/bin/env python
# coding: utf-8

# # Note
# This notebook is to train the encoder as a classifier with the idea of validate the encoder architecture first and then use this to train the VAE.

# # Parameters to experiment

# In[2]:


sizePercents = [0.1, 0.3, 0.5, 0.7, 1]

# to analuze execution times
useGpu = False
# useGpu = True

# trainingOnGuanaco = False
trainingOnGuanaco = True


# In[3]:


# training on guanaco
# ATENTION: if it is going to run on guanaco:
# 1) comment the %matplotlib magic in next block and any magic (something like %code)
# 2) Change to True the trainingOnGuanaco vairbale
# 3) set epoch with an appropiate number
# 4) add comment to experiemnts
# 5) Add this file as python file 
# 6) Change launchJobOnGuanaco file to run this file but with python format
# trainingOnGuanaco = False

# train without notebook
trainWithJustPython = False

# number_experiment (this is just a name)
# priors:
# 1
number_experiment = 99
number_experiment = str(number_experiment)

# seed to generate same datasets
seed = 0

# training
epochs = 3

# max elements by class
# max_elements_per_class = 15000 # this is the definitive in the work
max_elements_per_class = 15000

# train with previous model
trainWithPreviousModel = True

# include delta errors
includeDeltaErrors = True

# band
#passband = [5]
passband = [0, 1, 2, 3, 4, 5]


# include ohter feautures
includeOtherFeatures = True

# num of features to add
# ṕvar by channel
otherFeaturesDim = 12


# In[4]:


# cuda device
cuda_device = 0
cuda_device = "cuda:" + str(cuda_device)

# classes to analyze
# 42,  90,  16,  67,  62, 993,  92,  52,  88,  65, 991, 992,  15,
#        95,   6,  53, 994,  64

# periodic
# only_these_labels = [16, 92, 53]

# periodic + variable
only_these_labels = [16, 92, 53, 88, 65, 6]
# 53 has 24 light curves

# only_these_labels = [16, 92]
# only_these_labels = [16, 92]
# only_these_labels = [42,  90,  16,  67,  62, 993,  92,  52,  88,  65, 991, 992,  15,
#         95,   6,  53, 994,  64]

# VAE parameters
latentDim = 100
hiddenDim = 100
inputDim = 72

batch_training_size = 128

# early stopping 
threshold_early_stop = 1500


# In[5]:


# training params
learning_rate = 1e-4


# In[6]:


# add general comment about experiment 
# comment = "encoder as clasifier with periodic + variable (with class balancing) + 1 conv layer more"
comment = "exp " + number_experiment + " + encoder as clasifier with periodic + variable + class balancing + 1 conv layer more + " + str(len(passband)) + " channels + seed " + str(seed) + " + " + ("include delta errors" if includeDeltaErrors else "without delta errors") + " + max by class " + str(max_elements_per_class) + " + " + ("" if includeOtherFeatures else "not") + " other features"

print(comment)


# # Import libraries

# In[7]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils import data

# from tqdm import tqdm_notebook

if not trainingOnGuanaco:
    
    get_ipython().run_line_magic('matplotlib', 'notebook')
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
else:
    print("not load magics")
    
# import functions to load dataset
import sys
sys.path.append("./codesToDatasets")
from plasticc_dataset_torch import get_plasticc_datasets
from plasticc_plotting import plot_light_curve

import math

from torch import nn

# local imports
# %load_ext autoreload
# %autoreload 2
sys.path.append('../models')
# from classifier import EncoderClassifier, 
from classifierPrototype import EncoderClassifier

sys.path.append("./aux/")
from auxFunctions import *

from sklearn.model_selection import train_test_split


# ## Load the path to save model while training

# In[8]:


import os

# create experiment's folder
tmpGuanaco = "/home/lbravo/thesis/thesis/work/thesis/"
tmpLocal = "/home/leo/Desktop/thesis/work/thesis/"

# expPath = "experiments/" + number_experiment + "/seed" + str(seed) + "/maxClass" + str(int(max_elements_per_class/1000)) + "k"
# this si to use the 15k model but using other max of classes as dataset
# expPath = "experiments/" + number_experiment + "/seed" + str(seed) + "/maxClass" + str(15) + "k"
# 18 is the final model
expPath = "experiments/" + str(number_experiment) + "/seed" + str(seed) + "/maxClass" + str(15) + "k"

folder_path = (tmpGuanaco + expPath) if trainingOnGuanaco else (tmpLocal + expPath)


# check if folder exists
if not(os.path.isdir(folder_path)):
        
    # create folder
    try:
        os.makedirs(folder_path)
        
    except OSError as error:
        print ("Creation of the directory %s failed" % folder_path)
        print(error)
    else:
        print ("Successfully created the directory %s " % folder_path)
else:
    print("folder already exists")

    
    
# # define paht to save model while training
# pathToSaveModel = (tmpGuanaco + expPath + "/model") if trainingOnGuanaco else (tmpLocal + expPath + "/model")


# # Load data

# In[9]:


# define path to dataset
pathToFile = "/home/shared/astro/PLAsTiCC/" if trainingOnGuanaco else "/home/leo/Downloads/plasticData/"


# ## Loading dataset with pytorch tool

# In[10]:


# torch_dataset_lazy = get_plasticc_datasets(pathToFile)

# Light curves are tensors are now [bands, [mjd, flux, err, mask],
# lc_data, lc_label, lc_plasticc_id                              
torch_dataset_lazy = get_plasticc_datasets(pathToFile, only_these_labels=only_these_labels, max_elements_per_class = max_elements_per_class)


# # Spliting data (train/test)

# In[11]:


# splitting the data

# get light curves ids, targets
ids, targets, lightCurvesIds = getLightCurvesIds(torch_dataset_lazy)

# test array shapes
# assert len(targets) == torch_dataset_lazy.__len__()
# print(ids, len(ids), targets, len(targets))
# get light curves targets
print("# light curves ids: " + str(len(ids)))

# split training
trainIdxOriginal, tmpIdx = train_test_split(
    ids,
    test_size = 0.01,
    shuffle = True,
    stratify = targets,
    random_state = seed
)

# float to int
tmpIdx = tmpIdx.astype(int)

# split val, test
valIdx, testIdx = train_test_split(
    tmpIdx,
#     targets,
    test_size = 0.5,
    shuffle = True,
    stratify = targets[tmpIdx],
    random_state = seed
)

# float to int
trainIdxOriginal = trainIdxOriginal.astype(int)
valIdx = valIdx.astype(int)
testIdx = testIdx.astype(int)


# # define subsampling

# In[12]:


# len(trainIdx)


# In[13]:


# # sizePercent = 0.02

# finalIndex = int(len(trainIdx)*sizePercent)
# # print(finalIndex)

# trainIdx = trainIdx[0:finalIndex]


# In[14]:


# print(len(trainIdx))


# In[15]:


# saving ids
# this is becasue RF file uses it
# saveLightCurvesIdsBeforeBalancing(trainIdx, valIdx, testIdx, folder_path, lightCurvesIds, targets)


# ## Defining parameters to Autoencoder

# In[16]:


# check number of parameters
# latentDim = 5
# hiddenDim = 10
# inputDim = 72

latentDim = latentDim
hiddenDim = hiddenDim
inputDim = inputDim

# passband = passband

expPath_ = "experiments/18/seed" + str(seed) + "/maxClass" + str(15) + "k"
pathToSaveModel_ = (tmpGuanaco + expPath_ + "/model") if trainingOnGuanaco else (tmpLocal + expPath_ + "/model")


num_classes = len(only_these_labels)

if useGpu:
    
    print("gpu")
    
    # loading model
    model = torch.load(pathToSaveModel_ + ".txt").to(device = cuda_device)
    
    print("loading saved model")
    

else:
    print("cpu")
    model = torch.load(pathToSaveModel_ + ".txt").cpu()


# In[17]:


print(model)


# # Get own model predictions

# In[18]:


# %%timeit -n 1 -r 1
# iterate on test dataset
# for data_ in trainLoader:

def trainModelGPU(trainIdx):
    
    
    # training loader
    trainLoader = torch.utils.data.DataLoader(
        torch_dataset_lazy, 
        batch_size = 128, 
        # to balance classes
        sampler=ImbalancedDatasetSampler(
            torch_dataset_lazy, 
            indices = trainIdx,
            seed = seed
        ),
        # each worker retrieve data from disk, so the data will be ready to be processed by main process. The main process should get the data from disk, so if workers > 0, the workers will get the data (not the main process)
        num_workers = 1,

        # https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
        # the dataloader loads the data in pinned memory (instead of pageable memory), avoiding one process (to transfer data from pageable memory to pinned memory, work done by CUDA driver)
        pin_memory = True,
    )




    for idx, data_ in enumerate(trainLoader):

            # GPU
            data = data_[0].to(device = cuda_device)
            # print(data.get_device())
            data = generateDeltas(data.to(device = cuda_device), passband, includeOtherFeatures).type(torch.FloatTensor).to(device = cuda_device)

            # get model output
            outputs = model.forward(data, includeDeltaErrors)


# In[19]:


def trainModelCPU(trainIdx):
    
    
    # training loader
    trainLoader = torch.utils.data.DataLoader(
        torch_dataset_lazy, 
        batch_size = 128, 
        # to balance classes
        sampler=ImbalancedDatasetSampler(
            torch_dataset_lazy, 
            indices = trainIdx,
            seed = seed
        ),
        # each worker retrieve data from disk, so the data will be ready to be processed by main process. The main process should get the data from disk, so if workers > 0, the workers will get the data (not the main process)
        num_workers = 1,

        # https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
        # the dataloader loads the data in pinned memory (instead of pageable memory), avoiding one process (to transfer data from pageable memory to pinned memory, work done by CUDA driver)
        pin_memory = True,
    )




    for idx, data_ in enumerate(trainLoader):


            #CPU
            data = data_[0]
            data = generateDeltas(data, passband, includeOtherFeatures).type(torch.FloatTensor)

            # get model output
            outputs = model.forward(data, includeDeltaErrors)


# In[20]:


for sizePercent in sizePercents:
    
    
    # sizePercent = 0.02

    finalIndex = int(len(trainIdxOriginal)*sizePercent)
#     print(finalIndex)

    trainIdx = trainIdxOriginal[0:finalIndex]
    print(len(trainIdx))
    
    saveLightCurvesIdsBeforeBalancingForExecutionTimeAnalysis(trainIdx, valIdx, testIdx, folder_path, lightCurvesIds, targets, sizePercent)

    if useGpu: 
        
        get_ipython().run_line_magic('timeit', '-n 10 -r 10 trainModelGPU(trainIdx)')
        
    else:
        
        get_ipython().run_line_magic('timeit', '-n 10 -r 10 trainModelCPU(trainIdx)')

