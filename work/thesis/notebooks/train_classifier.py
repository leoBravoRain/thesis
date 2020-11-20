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
number_experiment = 8
number_experiment = str(number_experiment)

# add general comment about experiment 
# comment = "encoder as clasifier with periodic + variable (with class balancing) + 1 conv layer more"
comment = "encoder as clasifier with periodic + variable (with class balancing) + 1 conv layer more + 6 channels"


# In[32]:


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

# training
epochs = 2

# band
# passband = 5
passband = [0, 1, 2, 3, 5]

batch_training_size = 128


# In[33]:


# training params
learning_rate = 1e-3


# # Import libraries

# In[4]:


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


# # Load data

# In[5]:


# define path to dataset
pathToFile = "/home/shared/astro/PLAsTiCC/" if trainingOnGuanaco else "/home/leo/Downloads/plasticc_torch-master/"


# ## Loading dataset with pytorch tool

# In[6]:


# torch_dataset_lazy = get_plasticc_datasets(pathToFile)

# Light curves are tensors are now [bands, [mjd, flux, err, mask],
# lc_data, lc_label, lc_plasticc_id                              
torch_dataset_lazy = get_plasticc_datasets(pathToFile, only_these_labels=only_these_labels)


# # Spliting data (train/test)

# In[7]:


# get_ipython().run_line_magic('pinfo', 'torch.utils.data.random_split')


# In[19]:


# Spliting the data

# print(torch_dataset_lazy.__len__())

# selecting train splitting
train_size = int(0.8 * torch_dataset_lazy.__len__())
#print(train_size)

# getting test splitting
validation_size = math.floor((torch_dataset_lazy.__len__() - train_size)/2)
#print(validation_size)

# getting test splitting
test_size = torch_dataset_lazy.__len__() - train_size - validation_size
#print(test_size)

# spliting the torch dataset
# set seed
# torch.manual_seed(0)
trainDataset, validationDataset,  testDataset = torch.utils.data.random_split(
    torch_dataset_lazy, 
    [train_size, validation_size, test_size],
    generator = torch.Generator().manual_seed(0)
#     generator=torch.manual_seed(0),
)

print("train size:", train_size)
print("validation size: ", validation_size)
print("test size:", test_size)
print("sum: ", train_size+ validation_size + test_size)


# ## Create a dataloader

# In[20]:


print("initila distribution")
initialClassesDistribution = countClasses(trainDataset, only_these_labels)

# fig, ax = plt.subplots()
# ax.bar(x = np.arange(len(only_these_labels)), height = initialClassesDistribution)


# In[21]:


# # Create data loader (minibatches)

# training loader
trainLoader = torch.utils.data.DataLoader(
    trainDataset, 
    batch_size = batch_training_size, 
    # to balance classes
    sampler=ImbalancedDatasetSampler(trainDataset),
)

# validation loader
validationLoader = torch.utils.data.DataLoader(validationDataset, batch_size= batch_training_size,  num_workers = 4)

# # test loader
testLoader = torch.utils.data.DataLoader(testDataset)
# trainLoader = torch.utils.data.DataLoader(torch_dataset_lazy, batch_size=256, shuffle=True, num_workers=0)


# In[22]:


print("balanced distribution")
balancedClassesDistribution = countClasses(trainLoader, only_these_labels)

# fig, ax = plt.subplots()
# ax.bar(x = np.arange(6), height = balancedClassesDistribution)
# ax.bar(x = only_these_labels, height = temp2, width = 10)


# In[23]:


# firstIds = [] 

# for data in trainLoader:
    
#     # ids: data[2]
#     firstIds.extend(data[2].tolist())

# # print(len(firstIds))
# # print(len(secondIds))


# In[24]:


# secondIds = []

# for data in trainLoader:
    
#     # ids: data[2]
#     secondIds.extend(data[2].tolist())

# # print(len(firstIds))
# # print(len(secondIds))


# In[25]:


# test id arrays is the same
# firstIds.sort()
# secondIds.sort()


# prevIds = np.array(firstIds)

# currentIds = np.array(secondIds)

# print(prevIds[:4])
# print(currentIds[:4])
# comp = prevIds == currentIds

# # print(comp)

# # print(comp.all())
# # print(prevIds == currentIds)
# assert (comp).all(), "Should be the same ids"
# print("test ok")


# ## Load the path to save model while training

# In[26]:


import os

# create experiment's folder
folder_path = ("/home/lbravo/thesis/work/thesis/experiments/" + number_experiment) if trainingOnGuanaco else ("/home/leo/Desktop/thesis/work/thesis/experiments/" + number_experiment)
# !mkdir folder_path
# os.makedirs(os.path.dirname(folder_path), exist_ok=True)

# check if folder exists
if not(os.path.isdir(folder_path)):
        
    # create folder
    try:
        os.mkdir(folder_path)
    except OSError:
        print ("Creation of the directory %s failed" % folder_path)
    else:
        print ("Successfully created the directory %s " % folder_path)
else:
    print("folder already exists")

# define paht to save model while training
pathToSaveModel = "/home/lbravo/thesis/thesis/work/thesis/experiments/" + number_experiment + "/model" if trainingOnGuanaco else "/home/leo/Desktop/thesis/work/thesis/experiments/" + number_experiment + "/model"


# In[27]:


# store varibales on file
text_file = open("../experiments/" + number_experiment + "/experimentParameters.txt", "w")
text = "N° experiment: {7}\n General comment: {13}\n Classes: {0}\n train_size: {9}\n validation_size: {10}\n test_size: {11}\n total dataset size: {12}\n Epochs: {8}\n Latent dimension: {1}\n Hidden dimension: {2}\n Input dimension: {3}\n Passband: {4}\n Learning rate: {5}\n Batch training size: {6}\n initial train classes distribution: {14}\nbalanced train class distribution: {15}".format(only_these_labels, latentDim, hiddenDim, inputDim, passband, learning_rate, batch_training_size, number_experiment, epochs, train_size, validation_size, test_size, train_size + validation_size + test_size, comment, initialClassesDistribution, balancedClassesDistribution)
text_file.write(text)
text_file.close()
print("experiment parameters file created")


# ## Defining parameters to Autoencoder

# In[28]:


# check number of parameters
# latentDim = 5
# hiddenDim = 10
# inputDim = 72

latentDim = latentDim
hiddenDim = hiddenDim
inputDim = inputDim

# passband = passband

num_classes = len(only_these_labels)


# defining model
model = EncoderClassifier(latent_dim = latentDim, hidden_dim = hiddenDim, input_dim = inputDim, num_classes = num_classes, passband = passband)

# mdel to GPU
model = model.cuda()


# In[29]:


print(model)


# ### Training

# In[30]:


import torch
print(torch.__version__)


# In[31]:


from sklearn.metrics import f1_score

# optimizera
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

# # # loss plot
# if it is not cluster
if (not trainingOnGuanaco) or (not trainWithJustPython):
    
    # add f1 and loss plots
    fig, ax = plt.subplots(1, 2, figsize = (7, 3), tight_layout = True)
    # # fig, ax = plt.subplots()
    
    # error
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Error")
    
    
    # f1 score
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("F1 score")
    

# early stopping
prior_test_error = 0
count_early_stop = 0
threshold_early_stop = 20


print("starting the training")


# epoch
for nepoch in range(epochs):
        
    print("epoch:    {0} / {1}".format(nepoch, epochs))
    
    
    
     
    ######## Train ###########
    epoch_train_loss = 0
    
    for data_ in trainLoader:
        
        data = data_[0]
        labels = data_[1].cuda()
        
        optimizer.zero_grad()
            
        # this take the deltas (time and magnitude)
#         data = generateDeltas(data, passband).type(torch.FloatTensor).cuda()
        data = generateDeltas(data, passband).type(torch.FloatTensor).cuda()

#         # testing tensor size 
#         assert data.shape == torch.Size([batch_training_size, len(passband), 4, 71]), "Shape should be [minibatch size, channels, 4, 71]"
#         print("test ok")
        
        # get model output
        outputs = model.forward(data, passband)
        
#         # testing output shape size
#         assert outputs.shape == torch.Size([batch_training_size, len(only_these_labels)]), "Shape should be [minibatch, classes]"
#         print("test ok")

        # loss function
        loss = lossFunction(outputs, mapLabels(labels, only_these_labels).cuda())
        
        # backpropagation
        loss.backward()
        
        # update parameters
        optimizer.step()
        
        # add loss value (of the currrent minibatch)
        epoch_train_loss += loss.item()
        

    # get epoch loss value
    train_loss[nepoch] = epoch_train_loss / train_size
    
    
    
    
    ##### Validation ########
    
    epoch_test_loss = 0
    
    # check f1 score in each minibatch
    f1Score = 0
    
    batchCounter = 0
    
    # minibatches
    for data_ in validationLoader:
        
        data = data_[0]
        labels = data_[1].cuda()
        
        data = generateDeltas(data, passband).type(torch.FloatTensor).cuda()
        
        outputs = model.forward(data, passband)
        
#           # testing output shape size
#         assert outputs.shape == torch.Size([batch_training_size, len(only_these_labels)]), "Shape should be [minibatch, classes]"
#         print("test ok")

        # loss function
        loss = lossFunction(outputs, mapLabels(labels, only_these_labels).cuda())
    
        #  store minibatch loss value
        epoch_test_loss += loss.item()
        
        # f1 score
        f1Score += f1_score(mapLabels(labels, only_these_labels).cpu().numpy(), torch.argmax(outputs, 1).cpu().numpy(), average = "micro")
        
        # batch counter
        batchCounter += 1
    
    # get epoch test loss value
    test_loss[nepoch] = epoch_test_loss / validation_size
    
    # get epoch f1 score
    f1Scores[nepoch] = f1Score / batchCounter
    
    
    
    
    # plot loss values
    # if it's not cluster
    if (not trainingOnGuanaco) or (not trainWithJustPython):

        # loss values
        ax[0].plot(train_loss[0: nepoch], label = "train", linewidth = 3, c = "red") 
        ax[0].plot(test_loss[0: nepoch], label = "test", linestyle = "--", linewidth = 3, c = "green")
        
        # f1 score values
        ax[1].plot(f1Scores[0: nepoch], linewidth = 3, c = "green")
        
        # plot
        fig.canvas.draw()
    
    
    #### Early stopping #####
    
    
    
    # if new test loss is greater than the older one
    count_early_stop += 1
    if epoch_test_loss > prior_test_error:
        count_early_stop += 1
        print("early stopping counter: ", count_early_stop)
    else: 
        count_early_stop = 0
    
    # update prior test error
    prior_test_error = epoch_test_loss
    
    # analyze early stopping
    if count_early_stop > threshold_early_stop:
        
        print("Early stopping in epoch: ", nepoch)
        text_file = open("../experiments/" + number_experiment + "/earlyStopping.txt", "w")
        metricsText = "Epoch: {0}\n ES counter: {1}\n, Reconstruction test error: {2}".format(nepoch, count_early_stop, epoch_test_loss)
        text_file.write(metricsText)
        text_file.close()
        break
        
        
        
    #### Saving best model ####
    
    # if epoch test loss is smaller than global min
    if test_loss[nepoch] < minTestLossGlobalSoFar:
        
        # update global min
        minTestLossGlobalSoFar = test_loss[nepoch]
        
        # save model
        saveBestModel(model, pathToSaveModel, number_experiment, nepoch, minTestLossGlobalSoFar)
                
   


    # save losses
    print("saving losses")
    losses = np.asarray([train_loss, test_loss]).T
    np.savetxt("../experiments/" + number_experiment + "/training_losses.csv", losses, delimiter=",")
    

    
    
    # save f1 scores
    print("saving f1 scores")
    np.savetxt("../experiments/" + number_experiment + "/f1Scores.csv", f1Scores, delimiter=",")

    
    
# final message
print("training has finished")


# In[18]:


# get metrics on trainig dataset
getConfusionAndClassificationReport(trainLoader, nameLabel = "Train", passband = passband, model = model, staticLabels = only_these_labels, number_experiment = number_experiment)


# get metrics on validation dataset
getConfusionAndClassificationReport(validationLoader, nameLabel = "Validation", passband = passband, model = model, staticLabels = only_these_labels, number_experiment = number_experiment)


# ### Stop execution if it's on cluster

# In[19]:


import sys

if  trainingOnGuanaco or trainWithJustPython:

    sys.exit("Exit from code, because we are in cluster or running locally. Training has finished.")


# # Analyzing training

# In[15]:


get_ipython().system('cat ../experiments/8/experimentParameters.txt')


# In[16]:


# load losses array
losses = pd.read_csv("/home/leo/Desktop/thesis/work/thesis/experiments/"+ number_experiment + "/training_losses.csv")

# f1 scores
f1Scores = pd.read_csv("/home/leo/Desktop/thesis/work/thesis/experiments/"+ number_experiment + "/f1Scores.csv")

# plot losses
fig, ax = plt.subplots(1, 2, figsize = (10,4), tight_layout = True)

# loss
ax[0].set_xlabel("N° epoch")
ax[0].set_ylabel("Loss")
ax[0].plot(losses.iloc[:, 0], label = "train")
ax[0].plot(losses.iloc[:, 1], label = "validation")
ax[0].legend()

# f1 scores
ax[1].set_xlabel("N° epoch")
ax[1].set_ylabel("F1 score")
ax[1].plot(f1Scores)

# best model
# values copied from the txt file
# bestModelEpoch = 785
# bestModelError = 0.00434128265165168
# ax[0].scatter(bestModelEpoch, bestModelError, c = "r", linewidths = 10)
# ax[1].scatter(bestModelEpoch, f1Scores.iloc[bestModelEpoch], c = "r", linewidths = 10)


# In[17]:


get_ipython().system('cat ../experiments/8/bestScoresModelTraining.txt')


# In[23]:


# confusion matrix
import pandas as pd
import seaborn as sn

# get confusion matrix
cmTrain = pd.read_csv('../experiments/' + number_experiment + '/confusionMatrixTrain.csv', header = None) 
cmValidation = pd.read_csv('../experiments/' + number_experiment + '/confusionMatrixValidation.csv', header = None) 

print("Training")
sn.heatmap(cmTrain, annot=True)


# In[24]:


print("Validation")
sn.heatmap(cmValidation, annot = True)


# In[25]:


# classification report
get_ipython().system('cat ../experiments/8/clasificationReportTrain.txt')


# In[26]:


# classification report
get_ipython().system('cat ../experiments/8/clasificationReportValidation.txt')

