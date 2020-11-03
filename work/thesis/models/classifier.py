from torch import nn
import torch

# implementacion adaptada a 1D de https://github.com/naoto0804/pytorch-inpainting-with-partial-conv

class PartialConv(nn.Module):
    def __init__(self, in_channels_C,in_channels_M, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv1d(in_channels_C, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv1d(in_channels_M, out_channels, kernel_size,
                                   stride, padding, dilation, groups, False)
        # self.input_conv.apply(weights_init('kaiming'))

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self,input, mask):
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)
        #print(input.shape, mask.shape)
        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1).expand_as(output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)

        return output, new_mask
        
        

# building classifier

# encoder
class EncoderClassifier(torch.nn.Module):
    

    # init method
    def __init__(self, latent_dim, hidden_dim, input_dim, num_classes):
    
    
        super(type(self), self).__init__()
        
        # 1 Convolution layer
        # Conv1d(input channel, output channel, kernel size)
#         self.conv1 = torch.nn.Conv1d(1,64,3)
#         self.conv1 = torch.nn.Conv1d(1,64,3, stride = 2)
        
        # partial convolution
        self.pconv1 = PartialConv(in_channels_C = 1,in_channels_M = 1, out_channels = 64, kernel_size = 3, stride=2, padding=0, dilation=1, bias=True)
        
        # 2 Convolution layer
        # Conv1d(input channel, output channel, kernel size)
#         self.conv2 = torch.nn.Conv1d(64, 32, 3)
#         self.conv2 = torch.nn.Conv1d(64, 32, 3, stride = 2)
        
        # partial convolution
        self.pconv2 = PartialConv(in_channels_C = 64,in_channels_M = 64, out_channels = 32, kernel_size = 3, stride=2, padding=0, dilation=1, bias=True)
        
        # linear layer
#         self.hidden1 = torch.nn.Linear(2144*2, hidden_dim)
#         self.hidden1 = torch.nn.Linear(1088, hidden_dim)
        self.hidden1 = torch.nn.Linear(1632, hidden_dim)
        
#         self.hidden2 = torch.nn.Linear(hidden_dim, hidden_dim)
        
        # output layer
        self.outputLayer = torch.nn.Linear(hidden_dim, num_classes)
        
        # activation function
        self.activationConv = torch.nn.ReLU() #max(0, x)
#         self.activationConv = torch.nn.Tanh()
    
        # this works well.(comparing with relu)
        self.activationLinear = torch.nn.Tanh()

        # this is getting nan values
#         self.activationLinear = torch.nn.ReLU()

    # forward method
    def forward(self, x):
        
        # input shape: [batch_size, channels, sequence_length]
        # print("input shape: {0}".format(x.shape))
#         print("input to encoder: ")
#         print(x.shape)
        
        # convolution 1
        # x -> conv -> act -> ouput
        # shape should be: [batch_size, number of ouput channels (64), length of output from convolution]
        
        #conv to time
        # normal convolution
#         outputTimeConv = self.activationConv(self.conv1Time(x[:, 0, :].unsqueeze(1)))
#         outputTimeConv = self.activationConv(self.conv1(x[:, 0, :].unsqueeze(1)))
        # partial conv
        # output, newMask = pconv1(data, mask)
        outputTimeConv, maskTime = self.pconv1(x[:, 0, :].unsqueeze(1), x[:, 3, :].unsqueeze(1))
        # activation function
        outputTimeConv = self.activationConv(outputTimeConv)
        
        
        # conv to magnitude
#         outputMagConv = self.activationConv(self.conv1Mag(x[:, 1, :].unsqueeze(1)))
#         outputMagConv = self.activationConv(self.conv1(x[:, 1, :].unsqueeze(1)))
        
        # partial conv
        # output, newMask = pconv1(data, mask)
        outputMagConv, maskMag = self.pconv1(x[:, 1, :].unsqueeze(1), x[:, 3, :].unsqueeze(1))
        # activation function
        outputMagConv = self.activationConv(outputMagConv)
        
        
        # conv to mag error
#         outputMagErrorConv = self.activationConv(self.conv1(x[:, 2, :].unsqueeze(1)))
        
        # partial conv
        # output, newMask = pconv1(data, mask)
        outputMagErrorConv, maskError = self.pconv1(x[:, 2, :].unsqueeze(1), x[:, 3, :].unsqueeze(1))
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
#         outputMagErrorConv = self.activationConv(self.conv2(outputMagErrorConv))
        # partial conv
        outputMagErrorConv, maskError = self.pconv2(outputMagErrorConv, maskError)
        outputMagErrorConv = self.activationConv(outputMagErrorConv)
        
#         print("output conv2 shape: {0}".format(outputTimeConv.shape))
#         print("output conv2 shape: {0}".format(outputMagConv.shape))
        
        # flatten ouput
        # shape should be: [batch_size, -1]
        outputMagConv = outputMagConv.view(outputMagConv.shape[0], -1)
        
        outputTimeConv = outputTimeConv.view(outputTimeConv.shape[0], -1)
        
        outputMagErrorConv = outputMagErrorConv.view(outputMagErrorConv.shape[0], -1)
        
#         print("output reshape: ", outputMagConv.shape)
#         print("output reshape: ", outputTimeConv.shape)
                
        # concatenate 3 towers
#         output = torch.cat((outputMagConv, outputTimeConv), 1)
        output = torch.cat((outputTimeConv, outputMagConv, outputMagErrorConv), 1)
#         print("concatenate output shape: ", output.shape)
        
        # x -> hidden1 -> activation
#         print("before linear layer: {0}".format(output.shape))
        output = self.activationLinear(self.hidden1(output))
        # Should be an activiation function here?
#         output = (self.hidden1(output))
        
        output = self.outputLayer(output)
        
        # this should return the classification
        return output
