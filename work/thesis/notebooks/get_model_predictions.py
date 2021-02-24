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
number_experiment = 17
number_experiment = str(number_experiment)

# seed to generate same datasets
seed = 0

# training
epochs = 100000

# max elements by class
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
threshold_early_stop = 1500


# In[3]:


# training params
learning_rate = 1e-4


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

# In[6]:


import os

# create experiment's folder
tmpGuanaco = "/home/lbravo/thesis/thesis/work/thesis/"
tmpLocal = "/home/leo/Desktop/thesis/work/thesis/"

expPath = "experiments/" + number_experiment + "/seed" + str(seed) + "/maxClass" + str(int(max_elements_per_class/1000)) + "k"

folder_path = (tmpGuanaco + expPath) if trainingOnGuanaco else (tmpLocal + expPath)
# !mkdir folder_path
# os.makedirs(os.path.dirname(folder_path), exist_ok=True)

# # check if folder exists
# if not(os.path.isdir(folder_path)):
        
#     # create folder
#     try:
#         os.makedirs(folder_path)
        
#     except OSError as error:
#         print ("Creation of the directory %s failed" % folder_path)
#         print(error)
#     else:
#         print ("Successfully created the directory %s " % folder_path)
# else:
#     print("folder already exists")

# define paht to save model while training
pathToSaveModel = (tmpGuanaco + expPath + "/model") if trainingOnGuanaco else (tmpLocal + expPath + "/model")


# In[7]:


folder_path


# # Load data

# In[8]:


# define path to dataset
pathToFile = "/home/shared/astro/PLAsTiCC/" if trainingOnGuanaco else "/home/leo/Downloads/plasticData/"


# ## Loading dataset with pytorch tool

# In[9]:


# torch_dataset_lazy = get_plasticc_datasets(pathToFile)

# Light curves are tensors are now [bands, [mjd, flux, err, mask],
# lc_data, lc_label, lc_plasticc_id                              
torch_dataset_lazy = get_plasticc_datasets(pathToFile, only_these_labels=only_these_labels, max_elements_per_class = max_elements_per_class)


# In[10]:


assert torch_dataset_lazy.__len__() != 494096, "dataset should be smaller"
print("dataset test ok")


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


# In[12]:


# saving ids
# saveLightCurvesIdsBeforeBalancing(trainIdx, valIdx, testIdx, folder_path, lightCurvesIds, targets)


# In[13]:


# # load ids dictionary
# a_file = open(folder_path + "/dataset_ids_before_balancing.pkl", "rb")
# output = pickle.load(a_file)
# print(output)


# In[14]:


# # analize classes distributino
# fig, ax = plt.subplots(3, 1)

# ax[0].hist(targets[trainIdx])
# ax[1].hist(targets[valIdx])
# ax[2].hist(targets[testIdx])


# In[15]:


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

# In[16]:


# print("initila distribution")
# # initialClassesDistribution = countClasses(trainDataset, only_these_labels)
# initialClassesDistribution = np.unique(targets, return_counts=True)[1]

# print(initialClassesDistribution)

# # fig, ax = plt.subplots()
# # ax.bar(x = np.arange(len(only_these_labels)), height = initialClassesDistribution)


# In[17]:


# # Create data loader (minibatches)

# training loader
trainLoader = torch.utils.data.DataLoader(
    torch_dataset_lazy, 
#     batch_size = batch_training_size, 
    # to balance classes
    sampler=ImbalancedDatasetSampler(
        torch_dataset_lazy, 
        indices = trainIdx,
        seed = seed
#         indices = [0, 1, 2]
    ),
#     sampler = torch.utils.data.SubsetRandomSampler(
#         trainIdx,
#         generator = torch.Generator().manual_seed(seed)
#     ),
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
#     batch_size= batch_training_size,  
    num_workers = 4,
    pin_memory = True,
    sampler = valIdx,
#     sampler = torch.utils.data.SubsetRandomSampler(
#         valIdx,
#         generator = torch.Generator().manual_seed(seed)
#     ),
#     sampler=ImbalancedDatasetSampler(
#         torch_dataset_lazy, 
#         indices = valIdx,
#         seed = seed
# #         indices = [0, 1, 2]
#     ),
)

# # test loader
# testLoader = torch.utils.data.DataLoader(testDataset)
testLoader = torch.utils.data.DataLoader(
#     validationDataset, 
    torch_dataset_lazy,
#     batch_size= batch_training_size,  
    num_workers = 4,
    pin_memory = True,
    sampler = testIdx,
#     sampler = torch.utils.data.SubsetRandomSampler(
#         testIdx,
#         generator = torch.Generator().manual_seed(seed)
#     ),
)


# In[18]:


# # testIdx

# testIdAfterBalancing = np.zeros(shape = (test_size))

# for idx, data in enumerate(testLoader):
    
# #     print(data[2])
#     testIdAfterBalancing[idx] = data[2]
    
# # prin


# In[19]:


# assert np.array_equal(lightCurvesIds[testIdx], testIdAfterBalancing)


# In[20]:


# print("balanced distribution")
# balancedClassesDistribution = countClasses(trainLoader, only_these_labels)

# print(balancedClassesDistribution)
# # fig, ax = plt.subplots()
# # ax.bar(x = np.ar# return 0# return 0ange(6), height = balancedClassesDistribution)
# # ax.bar(x = only_these_labels, height = temp2, width = 10)


# In[21]:


# save ids of dataset to use (train, test and validation)
# saveLightCurvesIdsAfterBalancing(trainLoader, train_size, testLoader, test_size, validationLoader, validation_size, path = folder_path)


# In[22]:


# # load ids dictionary
# a_file = open(folder_path + "/dataset_ids_after_balancing.pkl", "rb")
# output = pickle.load(a_file)
# print(output["validation"])


# # Get other features

# In[23]:


if includeOtherFeatures:
    
    # save features
    trainOtherFeaturesArray = np.zeros(shape = (train_size, otherFeaturesDim))
    validOtherFeaturesArray = np.zeros(shape = (validation_size, otherFeaturesDim))
    testOtherFeaturesArray = np.zeros(shape = (test_size, otherFeaturesDim))

    print("starting to get the other features")

    trainLastIndex = 0
    validLastIndex = 0
    testLastIndex = 0
    
    
    for trainData_ in trainLoader:
        
        # get other features by batch
        # [batch size, features]
        trainOtherFeatures = getOtherFeatures(trainData_[0]).to(device = cuda_device)

        # indexation
        trainLastIndex_ = trainLastIndex + trainData_[0].shape[0]
        
        # save features in array indexing them
        trainOtherFeaturesArray[trainLastIndex : trainLastIndex_] = trainOtherFeatures.cpu().numpy()
            
        # update indexs
        trainLastIndex = trainLastIndex_
    
    # test size
    assert trainLastIndex == train_size
    
    for validData_ in validationLoader:
        
        # get other features by batch
        # [batch size, features]
        validOtherFeatures = getOtherFeatures(validData_[0]).to(device = cuda_device)

        # indexation
        validLastIndex_ = validLastIndex + validData_[0].shape[0]
        
        # save features in array indexing them
        validOtherFeaturesArray[validLastIndex : validLastIndex_] = validOtherFeatures.cpu().numpy()
            
        # update indexs
        validLastIndex = validLastIndex_
    
    
    # add test
    
    
    # test size
    assert validLastIndex == validation_size
    
    
    # test
    for testData_ in testLoader:
        
        # get other features by batch
        # [batch size, features]
        testOtherFeatures = getOtherFeatures(testData_[0]).to(device = cuda_device)

        # indexation
        testLastIndex_ = testLastIndex + testData_[0].shape[0]
        
        # save features in array indexing them
        testOtherFeaturesArray[testLastIndex : testLastIndex_] = testOtherFeatures.cpu().numpy()
            
        # update indexs
        testLastIndex = testLastIndex_
    
    
    # add test
    
    
    # test size
    assert testLastIndex_ == test_size
    
    
    print("finish to get other features")
    
    print("normalize features")
    # normalize features
    trainNormalizedFeatures = torch.from_numpy(normalizeOtherFeatures(trainOtherFeaturesArray)).type(torch.FloatTensor)
    validNormalizedFeatures = torch.from_numpy(normalizeOtherFeatures(validOtherFeaturesArray)).type(torch.FloatTensor)
    testNormalizedFeatures = torch.from_numpy(normalizeOtherFeatures(testOtherFeaturesArray)).type(torch.FloatTensor)
    
    # check nan values
    print(f"nan values train: {np.any(torch.isnan(trainNormalizedFeatures).cpu().numpy())}")
    print(f"nan values valid: {np.any(torch.isnan(validNormalizedFeatures).cpu().numpy())}")
    print(f"nan values valid: {np.any(torch.isnan(testNormalizedFeatures).cpu().numpy())}")


# ## Create experiment parameters file

# In[24]:


# # store varibales on file
# if trainingOnGuanaco or trainWithJustPython:
#     text_file = open("../" + expPath + "/experimentParameters.txt" , "w")
#     text = "N° experiment: {7}\n General comment: {13}\n Classes: {0}\n train_size: {9}\n validation_size: {10}\n test_size: {11}\n total dataset size: {12}\n Epochs: {8}\n Latent dimension: {1}\n Hidden dimension: {2}\n Input dimension: {3}\n Passband: {4}\n Learning rate: {5}\n Batch training size: {6}\n initial train classes distribution: {14}\nbalanced train class distribution: {15}".format(only_these_labels, latentDim, hiddenDim, inputDim, passband, learning_rate, batch_training_size, number_experiment, epochs, train_size, validation_size, test_size, train_size + validation_size + test_size, comment, initialClassesDistribution, balancedClassesDistribution)
#     text_file.write(text)
#     text_file.close()
#     print("experiment parameters file created")


# ## Defining parameters to Autoencoder

# In[25]:


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
    
# else:
    
#     # defining model
#     model = EncoderClassifier(
#         latent_dim = latentDim, 
#         hidden_dim = hiddenDim, 
#         input_dim = inputDim, 
#         num_classes = num_classes, 
#         passband = passband, 
#         includeDeltaErrors = includeDeltaErrors,
#         includeOtherFeatures = includeOtherFeatures,
#         otherFeaturesDim = otherFeaturesDim,
#     )

#     # mdel to GPU
#     model = model.to(device = cuda_device)
    
#     print("creating model with default parameters")


# In[26]:


print(model)


# # Get own model predictions

# # Train

# In[27]:


# class predictions
trainModelPredictions = np.zeros(shape = (train_size,))

# lc ids
trainIds = np.zeros(shape = (train_size,))

# test labels
trainLabels = np.zeros(shape = (train_size,))

print("getting predictions on train")

# index = 0
    
# this is for getting the other features
trainLastIndex = 0
    
# iterate on test dataset
# for data_ in trainLoader:
for idx, data_ in enumerate(trainLoader):
        
#         # index to include batch data
#         index_ = index + data_[0].shape[0]

        data = data_[0]

        # this take the deltas (time and magnitude)
        data = generateDeltas(data, passband, includeDeltaErrors).type(torch.FloatTensor).to(device = cuda_device)
            
#         # get model output
#         outputs = model.forward(data, includeDeltaErrors)
        
        # add other features
        # [batch size, features dim]
        if includeOtherFeatures:
            
            # index to include batch data
            trainLastIndex_ = trainLastIndex + data_[0].shape[0]

            # get only the normalized data from the batch (by indexation)
            otherFeatures = trainNormalizedFeatures[trainLastIndex : trainLastIndex_, :].to(device = cuda_device)
            
            # update index 
            trainLastIndex = trainLastIndex_
            
            # validate data
            if np.any(torch.isnan(otherFeatures).cpu().numpy()):
                
                print(f"other features with nan values in epoch {nepoch}")
            
            # get model output
            outputs = model.forward(data, includeDeltaErrors, otherFeatures)
            
            
        else:
            
            # get model output
            outputs = model.forward(data, includeDeltaErrors)
        
#         print(trainLastIndex_)
        
#         # get model predictions
#         trainModelPredictions[index : index_] = only_these_labels[torch.argmax(outputs, 1).cpu().numpy()[0]]
        
#         # get lc ids
#         trainIds[index : index_] = data_[2]
        
#         # save labels
#         trainLabels[index : index_] = data_[1]
        
#         # update index 
#         index = index_
        
        # get model predictions
        trainModelPredictions[idx] = only_these_labels[torch.argmax(outputs, 1).cpu().numpy()[0]]
        
        # get lc ids
        trainIds[idx] = data_[2]
        
        # save labels
        trainLabels[idx] = data_[1]
        
        
print("predictions ready")


# In[28]:


# # debugging
# print(np.unique(trainModelPredictions, return_counts=True)[0])

# print(np.unique(trainModelPredictions, return_counts=True)[1])


# In[29]:


# debugging

from sklearn.metrics import accuracy_score, f1_score

f1_score(
    trainLabels, 
    trainModelPredictions,
    average = "weighted"
)


# # Validation

# In[30]:


# class predictions
validModelPredictions = np.zeros(shape = (validation_size,))

# lc ids
validIds = np.zeros(shape = (validation_size,))

# test labels
validLabels = np.zeros(shape = (validation_size,))

print("getting predictions on validtion")

# index = 0
trainLastIndex = 0

# iterate on test dataset
# for data_ in (validationLoader):
for idx, data_ in enumerate(validationLoader):
    
#         # index to include batch data
#         index_ = index + data_[0].shape[0]
        
        data = data_[0]

        # this take the deltas (time and magnitude)
        data = generateDeltas(data, passband, includeDeltaErrors).type(torch.FloatTensor).to(device = cuda_device)
            
#         # get model output
#         outputs = model.forward(data, includeDeltaErrors)
        
        # add other features
        # [batch size, features dim]
        if includeOtherFeatures:
            
            # index to include batch data
            trainLastIndex_ = trainLastIndex + data_[0].shape[0]

            # get only the normalized data from the batch (by indexation)
            otherFeatures = validNormalizedFeatures[trainLastIndex : trainLastIndex_, :].to(device = cuda_device)
            
            # update index 
            trainLastIndex = trainLastIndex_
            
            # validate data
            if np.any(torch.isnan(otherFeatures).cpu().numpy()):
                
                print(f"other features with nan values in epoch {nepoch}")
            
            # get model output
            outputs = model.forward(data, includeDeltaErrors, otherFeatures)
            
            
        else:
            
            # get model output
            outputs = model.forward(data, includeDeltaErrors)
            
#         # get model predictions
#         validModelPredictions[index : index_] = only_these_labels[torch.argmax(outputs, 1).cpu().numpy()[0]]
        
#         # get lc ids
#         validIds[index : index_] = data_[2]
        
#         # save labels
#         validLabels[index : index_] = data_[1]
        
#         # update index 
#         index = index_

        # get model predictions
        validModelPredictions[idx] = only_these_labels[torch.argmax(outputs, 1).cpu().numpy()[0]]
        
        # get lc ids
        validIds[idx] = data_[2]
        
        # save labels
        validLabels[idx] = data_[1]
        
        
print("predictions ready")


# In[31]:


# print(np.unique(validModelPredictions, return_counts=True)[0])

# print(np.unique(validModelPredictions, return_counts=True)[1])


# In[32]:


# debugging

from sklearn.metrics import accuracy_score, f1_score

f1_score(
    validLabels, 
    validModelPredictions,
    average = "weighted"
)


# # Test

# In[33]:


# class predictions
testModelPredictions = np.zeros(shape = (test_size,))

# lc ids
testIds = np.zeros(shape = (test_size,))

# test labels
testLabels = np.zeros(shape = (test_size,))

print("getting predictions on test")

trainLastIndex = 0

# iterate on test dataset
for idx, data_ in enumerate(testLoader):
        
        data = data_[0]

        # this take the deltas (time and magnitude)
        data = generateDeltas(data, passband, includeDeltaErrors).type(torch.FloatTensor).to(device = cuda_device)
            
#         # get model output
#         outputs = model.forward(data, includeDeltaErrors)
        
        # add other features
        # [batch size, features dim]
        if includeOtherFeatures:
            
            # index to include batch data
            trainLastIndex_ = trainLastIndex + data_[0].shape[0]

            # get only the normalized data from the batch (by indexation)
            otherFeatures = testNormalizedFeatures[trainLastIndex : trainLastIndex_, :].to(device = cuda_device)
            
            # update index 
            trainLastIndex = trainLastIndex_
            
            # validate data
            if np.any(torch.isnan(otherFeatures).cpu().numpy()):
                
                print(f"other features with nan values in epoch {nepoch}")
            
            # get model output
            outputs = model.forward(data, includeDeltaErrors, otherFeatures)
            
            
        else:
            
            # get model output
            outputs = model.forward(data, includeDeltaErrors)
            
        # get model predictions
        testModelPredictions[idx] = only_these_labels[torch.argmax(outputs, 1).cpu().numpy()[0]]
        
        # get lc ids
        testIds[idx] = data_[2]
        
        # save labels
        testLabels[idx] = data_[1]
        
        
        
print("predictions ready")


# In[34]:


f1_score(
    testLabels, 
    testModelPredictions,
    average = "weighted"
)


# In[35]:


# print(trainIds[:3])
# print(trainLabels[:3])
# print(trainModelPredictions[:3])


# In[36]:


# print(validIds[:3])
# print(validLabels[:3])
# print(validModelPredictions[:3])


# In[37]:


# print(testIds[:3])
# print(testLabels[:3])
# print(testModelPredictions[:3])


# In[38]:


# save results
results = {
    
    # train
    "trainIds": trainIds,
    "trainLabels": trainLabels,
    "trainPredictions": trainModelPredictions,
    
     # validation
    "validIds": validIds,
    "validLabels": validLabels,
    "validPredictions": validModelPredictions,
    
    
    # test
    "testIds": testIds,
    "testLabels": testLabels,
    "testPredictions": testModelPredictions,
    
    
}

# save object

if trainingOnGuanaco or trainWithJustPython:

    a_file = open("../experiments/comparingModels/seed" + str(seed) + "/ownModel/OwnModel" + number_experiment + "Predictions.pkl", "wb")
    pickle.dump(results, a_file)
    a_file.close()

    print("model predictions saved on a file")
    
else:
    
    print("not save metrics")


# In[39]:


# load model
# a_file = open("../../experiments/comparingModels/seed" + str(seed) + "/ownModel/testOwnModelPredictions.pkl", "rb")
# output = pickle.load(a_file)
# print(output["testIds"].shape)


# ### Stop execution if it's on cluster

# In[40]:


import sys

if  trainingOnGuanaco or trainWithJustPython:

    sys.exit("Exit from code, because we are in cluster or running locally. Training has finished.")

