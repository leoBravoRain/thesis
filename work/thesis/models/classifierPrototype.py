from torch import nn
import torch

from partialConvolution import PartialConv
        
        

# building classifier

# encoder
class EncoderClassifier(torch.nn.Module):
    

    # init method
    def __init__(self, latent_dim, hidden_dim, input_dim, num_classes, passband, includeDeltaErrors = True, includeOtherFeatures = False, otherFeaturesDim = 0):
    
    
        super(type(self), self).__init__()
        
        # partial convolution
        self.pconv1 = PartialConv(in_channels_C = len(passband),in_channels_M = len(passband), out_channels = 64, kernel_size = 3, stride=2, padding=0, dilation=1, bias=True)
        
        # partial convolution
        self.pconv2 = PartialConv(in_channels_C = 64,in_channels_M = 64, out_channels = 32, kernel_size = 3, stride=2, padding=0, dilation=1, bias=True)
        
        # partial convolution
        self.pconv3 = PartialConv(in_channels_C = 32,in_channels_M = 32, out_channels = 32, kernel_size = 3, stride=2, padding=0, dilation=1, bias=True)
        
        # if add other features, so this layer will join the conv output and other features
        self.hidden1 = torch.nn.Linear(((768 + otherFeaturesDim) if includeOtherFeatures else 768) if includeDeltaErrors else 512, hidden_dim)
        
        # output layer
        self.outputLayer = torch.nn.Linear(hidden_dim, num_classes)
        
        # activation function
        #max(0, x)
        self.activationConv = torch.nn.ReLU()
    
        # this works well.(comparing with relu)
        self.activationLinear = torch.nn.Tanh()

#         # this is getting nan values
#         self.activationLinear = torch.nn.ReLU()
        
        # inlcude other features bool
        self.includeOtherFeatures = includeOtherFeatures

    # forward method
    # input shape: [batch_size, channels, sequence_length]
    def forward(self, x, includeDeltaErrors = True, otherFeatures = None):
        

        
        ################# convolution 1
        
        
        # x -> conv -> act -> ouput
        # shape should be: [batch_size, number of ouput channels (64), length of output from convolution]
        # this conv is applied to each channel, and then the ouput is for each output channel defined above.
        # the parameters have this shape: (output channel, input channel, other paramter i don't knwo what is? But it's same as kernel, but i think it's the shape of parameter and for coincidence is the kernel size (by the combination of other parameters values)
        
        
        #conv to time
        # partial conv
        # output, newMask = pconv1(data, mask)
        outputTimeConv, maskTime = self.pconv1(x[:, :, 0, :], x[:, :, -1, :])
        # activation function
        outputTimeConv = self.activationConv(outputTimeConv)
        
        
        # conv to magnitude
        # partial conv
        # output, newMask = pconv1(data, mask)
        outputMagConv, maskMag = self.pconv1(x[:, :, 1, :], x[:, :, -1, :])
        # activation function
        outputMagConv = self.activationConv(outputMagConv)
        
        
        # conv to mag error
        # partial conv
        # output, newMask = pconv1(data, mask)
        if includeDeltaErrors:
            
            outputMagErrorConv, maskError = self.pconv1(x[:, :, 2, :], x[:, :, -1, :])
            # activation function
            outputMagErrorConv = self.activationConv(outputMagErrorConv)
        
        
        
        
        ################# convolution 2
        
        
        
        # conv to time
        # partial conv
        outputTimeConv, maskTime = self.pconv2(outputTimeConv, maskTime)
        outputTimeConv = self.activationConv(outputTimeConv)
        
        
        # conv to flux
        # part conv
        outputMagConv, maskMag = self.pconv2(outputMagConv, maskMag)
        outputMagConv = self.activationConv(outputMagConv)
        
        # conv to mag error
        if includeDeltaErrors:
            
            # partial conv
            outputMagErrorConv, maskError = self.pconv2(outputMagErrorConv, maskError)
            outputMagErrorConv = self.activationConv(outputMagErrorConv)

        
        
        
        
        ################## convolution 3
        
        
        
        # conv to time
        # partial conv
        outputTimeConv, maskTime = self.pconv3(outputTimeConv, maskTime)
        outputTimeConv = self.activationConv(outputTimeConv)
        
        
        # conv to flux
        # part conv
        outputMagConv, maskMag = self.pconv3(outputMagConv, maskMag)
        outputMagConv = self.activationConv(outputMagConv)
        
        # conv to mag error
        if includeDeltaErrors:
            
            # partial conv
            outputMagErrorConv, maskError = self.pconv3(outputMagErrorConv, maskError)
            outputMagErrorConv = self.activationConv(outputMagErrorConv)
        
        
        
        
        
        ################## flatten conv ouput
        
        
        
        
        
        # shape should be: [batch_size, -1]
        outputMagConv = outputMagConv.view(outputMagConv.shape[0], -1)
        
        outputTimeConv = outputTimeConv.view(outputTimeConv.shape[0], -1)
        
        if includeDeltaErrors:
            
            outputMagErrorConv = outputMagErrorConv.view(outputMagErrorConv.shape[0], -1)
        
        
        
        
        
        ################## concatenate 3 towers + other features
        
        
        
        
        # the data can be with other features, w / o delta errors
        if includeDeltaErrors:
            
            if self.includeOtherFeatures:
                
                # concatenate 3 towers + features
                output = torch.cat((outputTimeConv, outputMagConv, outputMagErrorConv, otherFeatures), dim = 1)
                
            else:
                
                # concatenate 3 towers
                output = torch.cat((outputTimeConv, outputMagConv, outputMagErrorConv), dim = 1)
    
            
        else:
            
            # concatenate 2 towers
            output = torch.cat((outputTimeConv, outputMagConv), 1)
            
            
            
            
        ################## hidden 1
        
        
        
        
            
        # x -> hidden1 -> activation
        output = self.activationLinear(self.hidden1(output))
        
        
        
        
        ################## output layer
        
        
        
        
        # output
        output = self.outputLayer(output)
        
        
        
        
        # this should return the classification
        return output
