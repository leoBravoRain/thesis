# new comment
# import libraries
import numpy as np
import torch
# getting confusion amtrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pickle

# to compute features
from scipy.stats import chi2


# computing Pvar
def Pvar(magnitude, error):
    """
    Calculate the probability of a light curve to be variable.
    """

    #magnitude = data[0]
    #error = data[1]

#     print(magnitude)
    
#     print(error)
    
    mean_mag = np.mean(magnitude)
    nepochs = float(len(magnitude))

    chi = np.sum((magnitude - mean_mag)**2. / error**2.)
    p_chi = chi2.cdf(chi, (nepochs-1))

    return p_chi
    

# add other features
# data input: [ 128, 6, 4, 72 ] 
# data input: [batch, channels, [time, flux, err, mask], light curves samples]
def getOtherFeatures(data):
    
    # get means
    means = torch.zeros(size = (data.shape[0], data.shape[1]))
    
    # iq
    iqs = torch.zeros(size = (data.shape[0], data.shape[1]))
    
#     # pvar 
#     pvars = torch.zeros(size = (data.shape[0], data.shape[1]))
    
    # Each lc has a different lenght, so that's reason why it iterates over each channel and lc
    for lc_id in np.arange(data.shape[0]):
        
        for channel in np.arange(6):
            
            # get mask
            mask = data[lc_id, channel, 3, :].type(torch.BoolTensor)
            
            # filter light curve
            lc_masked = data[lc_id, channel, 1, mask]
            
#             # get pvars 
#             pvars[lc_id, channel] = Pvar(lc_masked.numpy(), data[lc_id, channel, 2, mask].numpy())
            
            # get means
            means[lc_id, channel] = torch.mean(lc_masked)
        
            # get IQ
            # output shape: [128, 6]
            # with small length values, the iq fails. It fails with length smaller or equal than 3
            if lc_masked.shape[0] > 3:
                
                iqs[lc_id, channel] = torch.kthvalue(lc_masked, int(0.75*lc_masked.shape[0]))[0] - torch.kthvalue(lc_masked, int(0.25*lc_masked.shape[0]))[0]

            #else:
                
                #print(f"lc smaller than 3. IQ value filled with 0")
                
    
    # concatenate data
    # data shape: [128, 12] == [batch, 6 channels means + 6 channels iq]
    concatenate = torch.tensor(np.concatenate((means, iqs), axis = 1))
#     concatenate = pvars
    
#     print(concatenate.shape)
    
    return concatenate

# save ids before balancing
def saveLightCurvesIdsBeforeBalancing(trainIdx, valIdx, testIdx, path, lightCurvesIds, labels):
    
    # get ids of light cur, ves
    ids = {
        "train": lightCurvesIds[trainIdx],
        "validation": lightCurvesIds[valIdx],
        "test": lightCurvesIds[testIdx],
        "message": "These are ids of light curves",
        "labels_train": labels[trainIdx],
        "labels_validation": labels[valIdx],
        "labels_test": labels[testIdx],
    }
    
    # save object
    a_file = open(path + "/dataset_ids_before_balancing.pkl", "wb")
    pickle.dump(ids, a_file)
    a_file.close()

    print("light curves ids saved on a file")
    
# save ids of dataset
def getIds(dataLoader, dataLoaderSize):

    idsArray = np.zeros(shape = (dataLoaderSize))
    labelsArray = np.zeros(shape = (dataLoaderSize))
    
    lastIndex = 0

    for idx, data in enumerate(dataLoader):

        lastIndex_ = lastIndex + data[0].shape[0]
        idsArray[lastIndex : lastIndex_] = data[2]
        
        labelsArray[lastIndex : lastIndex_] = data[1]

        lastIndex = lastIndex_

    return idsArray, labelsArray


# save object with ids of light curves
def saveLightCurvesIdsAfterBalancing(trainLoader, train_size, testLoader, test_size, validationLoader, validation_size, path):
    
    # get ids and labels
    trainIds, trainLabels = getIds(trainLoader, train_size)
    validIds, validLabels = getIds(validationLoader, validation_size)
    testIds, testLabels = getIds(testLoader, test_size)
    
    # get ids of light cur, ves
    ids = {

        "train": trainIds,
        "validation": validIds,
        "test": testIds,
        "message": "These are ids of light curves",
        "labels_train": trainLabels,
        "labels_validation": validLabels,
        "labels_test": testLabels,
    }
    
    #print(ids["train"].shape)
    #print(ids["labels_train"].shape)
    #print(ids["train"][:10])
    #print(ids["labels_train"][:10])
    
    # save object
    a_file = open(path + "/dataset_ids_after_balancing.pkl", "wb")
    pickle.dump(ids, a_file)
    a_file.close()
    
    print("light curves ids saved on a file")

# get light curves ids
def getLightCurvesIds(dataset):
    
    totalSize = dataset.__len__()
    
    ids = np.zeros(shape = (totalSize,))
    targets = np.zeros(shape = (totalSize,))
    
    lightCurvesIds = np.zeros(shape = (totalSize,))
    
    for idx, data in enumerate(dataset):
        
        targets[idx] = data[1]
        ids[idx] = idx
#         print(data[2])
        lightCurvesIds[idx] = data[2]
        
    return ids, targets, lightCurvesIds

# count classes in dataloader
# return array of counter of each class
def countClasses(dataLoader, labels):
    
    classCounter = np.zeros(shape = (len(labels),))
    
    for data in dataLoader:
        
        # count how many instance of class x
        for i in range(len(labels)):

            classCounter[i] += np.count_nonzero(data[1] == labels[i])
            
    return classCounter



# imbalanced dataset sampler
# This code was adapted from here: https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/165212
class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, seed, indices=None, num_samples=None, callback_get_label=None):

        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset 
        label_to_count = {}
        
        for idx in self.indices:
            label = self._get_label(dataset, idx)
#             label = 0]
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

        self.seed = seed
        
    def _get_label(self, dataset, idx):  
        
        # edit this for work with this dataset
        return dataset[idx][1]

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, 
            self.num_samples, 
            replacement=True,
            generator = torch.Generator().manual_seed(self.seed)
        ))

    def __len__(self):
        return self.num_samples
    
    
# it builds a mask for the deltas. It compares the next with the previous one element.
# original mask: [1,1, 0, 0]
# delta mask: [1, 0, 0]
# The same results is got with original_mask[:, 1:]
def generate_delta_mask(mask):
    
    # generate delta mask
#     mask_delta = mask[:, 1:].type(torch.BoolTensor) & mask[:, :-1].type(torch.BoolTensor)
    mask_delta = mask[:, :, 1:]
    
    return mask_delta


# function to generate delta time and flux
# data = [batchSize, channels, [time, flux, err, mask], light curve samples]
def generateDeltas(data, passBand,includeDeltaErrors = True):
    
    # work with delta time and magnitude
    
#     print("generate deltas input shape: {0}".format(data.shape) )
    # delta time
#     tmpDeltaTime = data[:, passBand, 0, 1:] - data[:, passBand, 0, :-1]
    tmpDeltaTime = data[:, passBand, 0, 1:] - data[:, passBand, 0, :-1]

#     print("generate deltas time shape: {0}".format(tmpDeltaTime.shape) )

#     # delta magnitude
    tmpDeltaMagnitude = data[:, passBand, 1, 1:] - data[:, passBand, 1, :-1]
#     print("generate deltas flux shape: {0}".format(tmpDeltaMagnitude.shape))
    
    # delta errors
    if includeDeltaErrors:
        
        tmpDeltaMagError = ((data[:, passBand, 2, 1:]**2) + (data[:, passBand, 2, :-1]**2))**0.5
        # tmpDeltaMagError = data[:, passBand, 2, 1:]
    
    # delta mask
    tmpMask = generate_delta_mask(data[:, passBand, 3,:])
    
    # concatenate tensors
    
    # if add delta errors
    if includeDeltaErrors:
        
        dataToUse = torch.cat((tmpDeltaTime.unsqueeze(2), tmpDeltaMagnitude.unsqueeze(2), tmpDeltaMagError.unsqueeze(2), tmpMask.unsqueeze(2)), 2)
        
    # if it does not add the delta errors
    else:
        
        dataToUse = torch.cat((tmpDeltaTime.unsqueeze(2), tmpDeltaMagnitude.unsqueeze(2), tmpMask.unsqueeze(2)), 2)
        
    #     print("data to use shape: {0}".format(dataToUse.shape))
    
    # normalize data
    # this was commented because it considerate that delta is already a normalization
#     dataToUse = normalizeLightCurve(dataToUse)
    
    # returning data
    return dataToUse


# mapping the labels to classes 0 to C-1

def mapLabels(labels, staticLabels):

    for i in range(len(staticLabels)):
        
        labels[labels == staticLabels[i]] = i 
        
    return labels


# save best model
def saveBestModel(model, pathToSaveModel, number_experiment, nepoch, newError, expPath):
    
    print("New min test loss. Saving model")

#         print("old: ", currentError)
#         print("new: ", newError)

    # save model
    # torch.save(state_dict(), pathToSaveModel)
    torch.save(model, pathToSaveModel + ".txt")

    # write metrics
    text_file = open("../" + expPath + "/bestScoresModelTraining.txt", "w")
    metricsText = "Epoch: {0}\n Reconstruction test error: {1}".format(nepoch, newError)
    text_file.write(metricsText)
    text_file.close()
    
    

# get confusion matrix and classification report
def getConfusionAndClassificationReport(dataSet, nameLabel, passband, model, staticLabels, number_experiment, expPath, includeDeltaErrors, includeOtherFeatures):
    
    # get y true and labels
    predictions = np.zeros(shape = (0,))
    labels_ = np.zeros(shape = (0,))

    # minibatches
    for data_ in dataSet:

        data = data_[0].cuda()
        labels = data_[1].cuda()

        data = generateDeltas(data, passband, includeDeltaErrors).type(torch.FloatTensor).cuda()
    
        if includeOtherFeatures:
            
            otherFeatures = getOtherFeatures(data_[0]).cuda()
        
        # get model output
        outputs = model.forward(data, includeDeltaErrors, otherFeatures)
        
        prediction = torch.argmax(outputs, 1).cpu().numpy()

        label = mapLabels(labels, staticLabels).cpu().numpy()

        predictions = np.append(predictions, prediction)
        labels_ = np.append(labels_, label)

    
    normalizes = ["true", "pred", "all"]
    for normalize in normalizes:
        
        cm = confusion_matrix(labels_, predictions, normalize = normalize)
        
        print("saving confusion matrix scores with normalize: " + normalize)
        
        np.savetxt("../" + expPath + "/confusionMatrix" + nameLabel + "_norm_" + normalize + ".csv", cm, delimiter=",")


    # classification report
    print("saving clasification report")
    text_file = open("../" + expPath + "/clasificationReport" + nameLabel + ".txt", "w")
    text = classification_report(labels_, predictions)
    text_file.write(text)
    text_file.close()
