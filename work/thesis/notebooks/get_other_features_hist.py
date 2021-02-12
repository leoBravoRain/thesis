#!/usr/bin/env python
# coding: utf-8

# # Note
# This notebook is to train the encoder as a classifier with the idea of validate the encoder architecture first and then use this to train the VAE.

# # Parameters to experiment

# In[1]:


# training on guanaco
# ATENTION: if it is going to run on guanaco:
# 1) comment the %matplotlib magic in next block and any magic (something like %code)
# 2) Change to True the trainingOnGuanaco vairbale
# 3) set epoch with an appropiate number
# 4) add comment to experiemnts
# 5) Add this file as python file 
# 6) Change launchJobOnGuanaco file to run this file but with python format
trainingOnGuanaco = True

# train without notebook
trainWithJustPython = False

# number_experiment (this is just a name)
# priors:
# 1
number_experiment = 14
number_experiment = str(number_experiment)

# seed to generate same datasets
seed = 0

# training
epochs = 1000

# max elements by class
max_elements_per_class = 15000

# train with previous model
trainWithPreviousModel = False

# include delta errors
includeDeltaErrors = True

# band
# passband = [5]
passband = [0, 1, 2, 3, 4, 5]


# include ohter feautures
includeOtherFeatures = True

# num of features to add
# ṕvar by channel
otherFeaturesDim = 12


# In[2]:


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
threshold_early_stop = 3000


# In[3]:


# training params
learning_rate = 1e-3


# In[4]:


# add general comment about experiment 
# comment = "encoder as clasifier with periodic + variable (with class balancing) + 1 conv layer more"
comment = "exp " + number_experiment + " + encoder as clasifier with periodic + variable + class balancing + 1 conv layer more + " + str(len(passband)) + " channels + seed " + str(seed) + " + " + ("include delta errors" if includeDeltaErrors else "without delta errors") + " + max by class " + str(max_elements_per_class) + " + " + ("" if includeOtherFeatures else "not") + " other features"

print(comment)


# # Import libraries

# In[5]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils import data

# from tqdm import tqdm_notebook

# %matplotlib notebook

# import functions to load dataset
import sys
sys.path.append("./codesToDatasets")
from plasticc_dataset_torch import get_plasticc_datasets
# from plasticc_plotting import plot_light_curve

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

# In[6]:


import os

# create experiment's folder
tmpGuanaco = "/home/lbravo/thesis/thesis/work/thesis/"
tmpLocal = "/home/leo/Desktop/thesis/work/thesis/"

expPath = "experiments/" + number_experiment + "/seed" + str(seed) + "/maxClass" + str(int(max_elements_per_class/1000)) + "k"

folder_path = (tmpGuanaco + expPath) if trainingOnGuanaco else (tmpLocal + expPath)
# !mkdir folder_path
# os.makedirs(os.path.dirname(folder_path), exist_ok=True)

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

# define paht to save model while training
pathToSaveModel = (tmpGuanaco + expPath + "/model") if trainingOnGuanaco else (tmpLocal + expPath + "/model")


# # Load data

# In[7]:


# define path to dataset
pathToFile = "/home/shared/astro/PLAsTiCC/" if trainingOnGuanaco else "/home/leo/Downloads/plasticData/"


# ## Loading dataset with pytorch tool

# In[8]:


# torch_dataset_lazy = get_plasticc_datasets(pathToFile)

# Light curves are tensors are now [bands, [mjd, flux, err, mask],
# lc_data, lc_label, lc_plasticc_id                              
torch_dataset_lazy = get_plasticc_datasets(pathToFile, only_these_labels=only_these_labels, max_elements_per_class = max_elements_per_class)


# In[9]:


assert torch_dataset_lazy.__len__() != 494096, "dataset should be smaller"
print("dataset test ok")


# # Spliting data (train/test)

# In[10]:


# splitting the data

# get light curves ids, targets
ids, targets, lightCurvesIds = getLightCurvesIds(torch_dataset_lazy)

# test array shapes
# assert len(targets) == torch_dataset_lazy.__len__()
# print(ids, len(ids), targets, len(targets))
# get light curves targets
print("# light curves ids: " + str(len(ids)))

# split training
trainIdx, tmpIdx = train_test_split(
    ids,
    test_size = 0.2,
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
trainIdx = trainIdx.astype(int)
valIdx = valIdx.astype(int)
testIdx = testIdx.astype(int)


# In[11]:


# saving ids
saveLightCurvesIdsBeforeBalancing(trainIdx, valIdx, testIdx, folder_path, lightCurvesIds, targets)


# In[12]:


# # load ids dictionary
# a_file = open(folder_path + "/dataset_ids_before_balancing.pkl", "rb")
# output = pickle.load(a_file)
# print(output)


# In[13]:


# # analize classes distributino
fig, ax = plt.subplots(3, 1)

ax[0].hist(targets[trainIdx])
ax[1].hist(targets[valIdx])
ax[2].hist(targets[testIdx])


# In[14]:


# # Spliting the data

# # print(torch_dataset_lazy.__len__())

totalSize = torch_dataset_lazy.__len__()

# totalSize = totalSize
# # print(totalSize)

# selecting train splitting
# train_size = int(0.8 * totalSize)
train_size = trainIdx.shape[0]
#print(train_size)

# # getting test splitting
# validation_size = math.floor((totalSize - train_size)/3)
validation_size = valIdx.shape[0]
# #print(validation_size)

# # getting test splitting
# test_size = totalSize - train_size - validation_size
test_size = testIdx.shape[0]
# #print(test_size)

# # spliting the torch dataset
# trainDataset, validationDataset,  testDataset = torch.utils.data.random_split(
#     torch_dataset_lazy, 
#     [train_size, validation_size, test_size],
    
#     # set seed
#     generator = torch.Generator().manual_seed(seed)
# )

print("train size:", train_size)
print("validation size: ", validation_size)
print("test size:", test_size)
totTmp = train_size+ validation_size + test_size
print("sum: ", totTmp)
assert torch_dataset_lazy.__len__() == totTmp, "dataset partition should be the same"


# ## Create a dataloader

# In[15]:


print("initila distribution")
# initialClassesDistribution = countClasses(trainDataset, only_these_labels)
initialClassesDistribution = np.unique(targets, return_counts=True)[1]

print(initialClassesDistribution)

# fig, ax = plt.subplots()
# ax.bar(x = np.arange(len(only_these_labels)), height = initialClassesDistribution)


# In[16]:


# # Create data loader (minibatches)

# training loader
trainLoader = torch.utils.data.DataLoader(
    torch_dataset_lazy, 
    batch_size = batch_training_size, 
    # to balance classes
    sampler=ImbalancedDatasetSampler(
        torch_dataset_lazy, 
        indices = trainIdx,
        seed = seed
#         indices = [0, 1, 2]
    ),
    # each worker retrieve data from disk, so the data will be ready to be processed by main process. The main process should get the data from disk, so if workers > 0, the workers will get the data (not the main process)
    num_workers = 4,
    
    # https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
    # the dataloader loads the data in pinned memory (instead of pageable memory), avoiding one process (to transfer data from pageable memory to pinned memory, work done by CUDA driver)
    pin_memory = True,
)


# validation loader
validationLoader = torch.utils.data.DataLoader(
#     validationDataset, 
    torch_dataset_lazy,
    batch_size= batch_training_size,  
    num_workers = 4,
    pin_memory = True,
    sampler = torch.utils.data.SubsetRandomSampler(
        valIdx,
        generator = torch.Generator().manual_seed(seed)
    ),
)

# # test loader
# testLoader = torch.utils.data.DataLoader(testDataset)
testLoader = torch.utils.data.DataLoader(
#     validationDataset, 
    torch_dataset_lazy,
#     batch_size= batch_training_size,  
    num_workers = 4,
    pin_memory = True,
    sampler = torch.utils.data.SubsetRandomSampler(
        testIdx,
        generator = torch.Generator().manual_seed(seed)
    ),
)


# In[17]:


print("balanced distribution")
balancedClassesDistribution = countClasses(trainLoader, only_these_labels)

print(balancedClassesDistribution)
# fig, ax = plt.subplots()
# ax.bar(x = np.ar# return 0# return 0ange(6), height = balancedClassesDistribution)
# ax.bar(x = only_these_labels, height = temp2, width = 10)


# In[18]:


# save ids of dataset to use (train, test and validation)
saveLightCurvesIdsAfterBalancing(trainLoader, train_size, testLoader, test_size, validationLoader, validation_size, path = folder_path)


# In[19]:


# # load ids dictionary
# a_file = open(folder_path + "/dataset_ids_after_balancing.pkl", "rb")
# output = pickle.load(a_file)
# print(output["validation"])


# ## Create experiment parameters file

# In[20]:


# store varibales on file
if trainingOnGuanaco or trainWithJustPython:
    text_file = open("../" + expPath + "/experimentParameters.txt" , "w")
    text = "N° experiment: {7}\n General comment: {13}\n Classes: {0}\n train_size: {9}\n validation_size: {10}\n test_size: {11}\n total dataset size: {12}\n Epochs: {8}\n Latent dimension: {1}\n Hidden dimension: {2}\n Input dimension: {3}\n Passband: {4}\n Learning rate: {5}\n Batch training size: {6}\n initial train classes distribution: {14}\nbalanced train class distribution: {15}".format(only_these_labels, latentDim, hiddenDim, inputDim, passband, learning_rate, batch_training_size, number_experiment, epochs, train_size, validation_size, test_size, train_size + validation_size + test_size, comment, initialClassesDistribution, balancedClassesDistribution)
    text_file.write(text)
    text_file.close()
    print("experiment parameters file created")


# ## Defining parameters to Autoencoder

# In[21]:


# check number of parameters
# latentDim = 5
# hiddenDim = 10
# inputDim = 72

latentDim = latentDim
hiddenDim = hiddenDim
inputDim = inputDim

# passband = passband

num_classes = len(only_these_labels)

if trainWithPreviousModel:
    
    # loadgin model
    model = torch.load(pathToSaveModel + ".txt").to(device = cuda_device)
    
    print("loading saved model")
    
else:
    
    # defining model
    model = EncoderClassifier(
        latent_dim = latentDim, 
        hidden_dim = hiddenDim, 
        input_dim = inputDim, 
        num_classes = num_classes, 
        passband = passband, 
        includeDeltaErrors = includeDeltaErrors,
        includeOtherFeatures = includeOtherFeatures,
        otherFeaturesDim = otherFeaturesDim,
    )

    # mdel to GPU
    model = model.to(device = cuda_device)
    
    print("creating model with default parameters")


# In[22]:


print(model)


# ### Training

# In[23]:


from sklearn.metrics import f1_score

# optimizeraa
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.5)

# loss function
lossFunction = nn.CrossEntropyLoss()

# loss
train_loss = np.zeros((epochs,))
test_loss = np.zeros((epochs,))

# f1 scores
f1Scores = np.zeros((epochs, ))

# min global test loss 
minTestLossGlobalSoFar = float("inf")


# early stopping
# prior_test_error = 0
count_early_stop = 0
threshold_early_stop = threshold_early_stop


# save features
otherFeaturesArray = np.zeros(shape = (train_size, otherFeaturesDim))

print("starting the training")


    
lastIndex = 0

for data_ in trainLoader:

    data = data_[0]
    labels = data_[1].to(device = cuda_device)
#         labels = data_[1]

    optimizer.zero_grad()

    # this take the deltas (time and magnitude)
    data = generateDeltas(data, passband, includeDeltaErrors).type(torch.FloatTensor).to(device = cuda_device)
#         data = generateDeltas(data, passband).type(torch.FloatTensor)

    # add other features
    # [batch size, features]
#         if includeOtherFeatures:
    if includeOtherFeatures:

        otherFeatures = getOtherFeatures(data_[0]).to(device = cuda_device)

        
        lastIndex_ = lastIndex + data_[0].shape[0]
        
        otherFeaturesArray[lastIndex : lastIndex_] = otherFeatures.cpu().numpy()
        
        lastIndex = lastIndex_
        
print("finish")


# # Save file

# In[24]:


# save array
a_file = open("./otherFeaturesValues.pkl", "wb")
pickle.dump(otherFeaturesArray, a_file)
a_file.close()


# ### Stop execution if it's on cluster

# In[25]:


import sys

if  trainingOnGuanaco or trainWithJustPython:

    sys.exit("Exit from code, because we are in cluster or running locally. Training has finished.")


# # Load

# In[26]:


# load
a_file = open("./otherFeaturesValues.pkl", "rb")
otherFeaturesArray = pickle.load(a_file)


# In[27]:


otherFeaturesArray.shape


# In[28]:


# assert (train_size, otherFeaturesDim) == otherFeaturesArray.shape

fig, ax = plt.subplots(otherFeaturesDim, 1, figsize = (5, 10))

for i in np.arange(otherFeaturesDim):
    
    ax[i].hist(otherFeaturesArray[:, i], bins = 100)

