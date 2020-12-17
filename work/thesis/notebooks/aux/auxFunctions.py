# import libraries
import numpy as np
import torch
# getting confusion amtrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# get light curves ids
def getLightCurvesIds(dataset):
    
    totalSize = dataset.__len__()
    
    ids = np.zeros(shape = (totalSize,))
    targets = np.zeros(shape = (totalSize,))
    
    for idx, data in enumerate(dataset):
        
        targets[idx] = data[1]
        ids[idx] = idx
    
    return ids, targets

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

    def __init__(self, dataset, indices=None, num_samples=None, callback_get_label=None):

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

    def _get_label(self, dataset, idx):  
        
        # edit this for work with this dataset
        return dataset[idx][1]

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

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
def generateDeltas(data, passBand):
    
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
    tmpDeltaMagError = data[:, passBand, 2, 1:] - data[:, passBand, 2, :-1]
    
    # delta mask
    tmpMask = generate_delta_mask(data[:, passBand, 3,:])
    
    # concatenate tensors
    dataToUse = torch.cat((tmpDeltaTime.unsqueeze(2), tmpDeltaMagnitude.unsqueeze(2), tmpDeltaMagError.unsqueeze(2), tmpMask.unsqueeze(2)), 2)
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
    torch.save(model.state_dict(), pathToSaveModel)

    # write metrics
    text_file = open("../" + expPath + "/bestScoresModelTraining.txt", "w")
    metricsText = "Epoch: {0}\n Reconstruction test error: {1}".format(nepoch, newError)
    text_file.write(metricsText)
    text_file.close()
    
    

# get confusion matrix and classification report
def getConfusionAndClassificationReport(dataSet, nameLabel, passband, model, staticLabels, number_experiment, expPath):
    
    # get y true and labels
    predictions = np.zeros(shape = (0,))
    labels_ = np.zeros(shape = (0,))

    # minibatches
    for data_ in dataSet:

        data = data_[0].cuda()
        labels = data_[1].cuda()

        data = generateDeltas(data, passband).type(torch.FloatTensor).cuda()

        outputs = model.forward(data)

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