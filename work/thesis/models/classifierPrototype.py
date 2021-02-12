from torch import nn
import torch

from partialConvolution import PartialConv
        
        

# building classifier

# encoder
class EncoderClassifier(torch.nn.Module):
    

    # init method
    def __init__(self, latent_dim, hidden_dim, input_dim, num_classes, passband, includeDeltaErrors = True, includeOtherFeatures = False, otherFeaturesDim = 0):
    
    
        super(type(self), self).__init__()
        
        # 1 Convolution layer
        # Conv1d(input channel, output channel, kernel size)
#         self.conv1 = torch.nn.Conv1d(1,64,3)
#         self.conv1 = torch.nn.Conv1d(1,64,3, stride = 2)
        
        # partial convolution
        self.pconv1 = PartialConv(in_channels_C = len(passband),in_channels_M = len(passband), out_channels = 64, kernel_size = 3, stride=2, padding=0, dilation=1, bias=True)
        
        # 2 Convolution layer
        # Conv1d(input channel, output channel, kernel size)
#         self.conv2 = torch.nn.Conv1d(64, 32, 3)
#         self.conv2 = torch.nn.Conv1d(64, 32, 3, stride = 2)
        
        # partial convolution
        self.pconv2 = PartialConv(in_channels_C = 64,in_channels_M = 64, out_channels = 32, kernel_size = 3, stride=2, padding=0, dilation=1, bias=True)
        
        # partial convolution
        self.pconv3 = PartialConv(in_channels_C = 32,in_channels_M = 32, out_channels = 32, kernel_size = 3, stride=2, padding=0, dilation=1, bias=True)
        
        # linear layer
#         self.hidden1 = torch.nn.Linear(2144*2, hidden_dim)
#         self.hidden1 = torch.nn.Linear(1088, hidden_dim)
#         self.hidden1 = torch.nn.Linear(1632, hidden_dim)
        
        self.hidden1 = torch.nn.Linear(((768 + otherFeaturesDim) if includeOtherFeatures else 768) if includeDeltaErrors else 512, hidden_dim)
#         self.hidden1 = torch.nn.Linear(((768) if includeOtherFeatures else 768) if includeDeltaErrors else 512, hidden_dim)
        
#         self.hidden2 = torch.nn.Linear(hidden_dim, hidden_dim)
        
        # output layer
        self.outputLayer = torch.nn.Linear(hidden_dim, num_classes)
#         self.outputLayer = torch.nn.Linear(hidden_dim + otherFeaturesDim, num_classes)
        
        # activation function
        self.activationConv = torch.nn.ReLU() #max(0, x)
#         self.activationConv = torch.nn.Tanh()
    
        # this works well.(comparing with relu)
        # self.activationLinear = torch.nn.Tanh()

        # this is getting nan values
        self.activationLinear = torch.nn.ReLU()
        
        
        self.includeOtherFeatures = includeOtherFeatures

    # forward method
    def forward(self, x, includeDeltaErrors = True, otherFeatures = None):
        
        # print(self.includeOtherFeatures)
        
        # input shape: [batch_size, channels, sequence_length]
#         print("input shape: {0}".format(x.shape))
#         print("input to encoder: ")
#         print(x.shape)
        
        # convolution 1
        # x -> conv -> act -> ouput
        # shape should be: [batch_size, number of ouput channels (64), length of output from convolution]
        # this conv is applied to each channel, and then the ouput is for each output channel defined above.
        # the parameters have this shape: (output channel, input channel, other paramter i don't knwo what is? But it's same as kernel, but i think it's the shape of parameter and for coincidence is the kernel size (by the combination of other parameters values)
        
        #conv to time
        # normal convolution
#         outputTimeConv = self.activationConv(self.conv1(x[:, 0, :].unsqueeze(1)))
        # partial conv
#         outputTimeConv, maskTime = self.pconv1(x[:, passband, 0, :].unsqueeze(1), x[:, passband, 3, :].unsqueeze(1))
        outputTimeConv, maskTime = self.pconv1(x[:, :, 0, :], x[:, :, -1, :])
        # activation function
        outputTimeConv = self.activationConv(outputTimeConv)
        
        
        # conv to magnitude
#         outputMagConv = self.activationConv(self.conv1Mag(x[:, 1, :].unsqueeze(1)))
#         outputMagConv = self.activationConv(self.conv1(x[:, 1, :].unsqueeze(1)))
        
        # partial conv
        # output, newMask = pconv1(data, mask)
        outputMagConv, maskMag = self.pconv1(x[:, :, 1, :], x[:, :, -1, :])
        # activation function
        outputMagConv = self.activationConv(outputMagConv)
        
        
        # conv to mag error
#         outputMagErrorConv = self.activationConv(self.conv1(x[:, 2, :].unsqueeze(1)))
        
        # partial conv
        # output, newMask = pconv1(data, mask)
        if includeDeltaErrors:
            
            outputMagErrorConv, maskError = self.pconv1(x[:, :, 2, :], x[:, :, -1, :])
            # activation function
            outputMagErrorConv = self.activationConv(outputMagErrorConv)
        
#         print("output conv1 shape: {0}".format(outputMagConv.shape))
#         print("output conv1 shape: {0}".format(outputTimeConv.shape))
        
        # convolution 2
#         # shape should be: [batch_size, number of ouput channels (32), length of output from convolution]
        
        
        # conv to time
#         outputTimeConv = self.activationConv(self.conv2(outputTimeConv))
        
        # partial conv
        outputTimeConv, maskTime = self.pconv2(outputTimeConv, maskTime)
        outputTimeConv = self.activationConv(outputTimeConv)
        
        
        # conv to flux
#         outputMagConv = self.activationConv(self.conv2(outputMagConv))
        # part conv
        outputMagConv, maskMag = self.pconv2(outputMagConv, maskMag)
        outputMagConv = self.activationConv(outputMagConv)
        
        # conv to mag error
        if includeDeltaErrors:
            
    #         outputMagErrorConv = self.activationConv(self.conv2(outputMagErrorConv))
            # partial conv
            outputMagErrorConv, maskError = self.pconv2(outputMagErrorConv, maskError)
            outputMagErrorConv = self.activationConv(outputMagErrorConv)

        
        # conv 3
        
        # partial conv
        outputTimeConv, maskTime = self.pconv3(outputTimeConv, maskTime)
        outputTimeConv = self.activationConv(outputTimeConv)
        
        
        # conv to flux
#         outputMagConv = self.activationConv(self.conv2(outputMagConv))
        # part conv
        outputMagConv, maskMag = self.pconv3(outputMagConv, maskMag)
        outputMagConv = self.activationConv(outputMagConv)
        
        # conv to mag error
        if includeDeltaErrors:
            
    #         outputMagErrorConv = self.activationConv(self.conv2(outputMagErrorConv))
            # partial conv
            outputMagErrorConv, maskError = self.pconv3(outputMagErrorConv, maskError)
            outputMagErrorConv = self.activationConv(outputMagErrorConv)
        
        
        
#         print("output conv2 shape: {0}".format(outputTimeConv.shape))
#         print("output conv2 shape: {0}".format(outputMagConv.shape))
        
        # flatten ouput
        # shape should be: [batch_size, -1]
        outputMagConv = outputMagConv.view(outputMagConv.shape[0], -1)
        
        outputTimeConv = outputTimeConv.view(outputTimeConv.shape[0], -1)
        
        if includeDeltaErrors:
            
            outputMagErrorConv = outputMagErrorConv.view(outputMagErrorConv.shape[0], -1)
        
#         print("output reshape: ", outputMagConv.shape)
#         print("output reshape: ", outputTimeConv.shape)
                
        # concatenate 3 towers + other features
#         output = torch.cat((outputMagConv, outputTimeConv), 1)
        if includeDeltaErrors:
            
#             print(outputTimeConv.shape, outputMagConv.shape,  outputMagErrorConv.shape, otherFeatures.shape)
            
            if self.includeOtherFeatures:
            
#                 print(otherFeatures.shape)
                
#                 print(otherFeatures)
                
                output = torch.cat((outputTimeConv, outputMagConv, outputMagErrorConv, otherFeatures), dim = 1)
            
            else:
                
                output = torch.cat((outputTimeConv, outputMagConv, outputMagErrorConv), dim = 1)
    
#             output = torch.cat((outputTimeConv, outputMagConv, outputMagErrorConv), dim = 1)
            
            
        else:
            
            output = torch.cat((outputTimeConv, outputMagConv), 1)
            
#         print("concatenate output shape: ", output.shape)
        
        
        # x -> hidden1 -> activation
#         print("before linear layer: {0}".format(output.shape))

        output = self.activationLinear(self.hidden1(output))
        
#         print("output hidden 1 ", output.shape)
        
#         if self.includeOtherFeatures:
            
# #                 print(otherFeatures.shape)
                
#                 output = torch.cat((output, otherFeatures), dim = 1)
        
#         print("output hidden 1 concatenated ", output.shape)
        # Should be an activiation function here?
#         output = (self.hidden1(output))
        
        output = self.outputLayer(output)
        
        # this should return the classification
        return output
