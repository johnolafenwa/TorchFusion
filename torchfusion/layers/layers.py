import torch.nn as nn
import torch
import torch.nn.functional as F
from torchfusion.initializers import *
from torch.nn.modules.conv import _ConvNd,_ConvTransposeMixin,_single,_pair,_triple
from torch.nn.modules.batchnorm import _BatchNorm


class ConvNd(_ConvNd):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias,out_padding,weight_init=Kaiming_Normal(),bias_init=Zeros(),activation=None):
        super(ConvNd,self).__init__(in_channels,out_channels,kernel_size,stride,padding,dilation,False,out_padding,groups,bias)
        self.activation = activation
        if weight_init is not None:
            weight_init(self.weight.data)
        if bias and bias_init is not None:
            bias_init(self.bias.data)

    def __call__(self, *args, **kwargs):
        outputs = super(ConvNd,self).__call__(*args,**kwargs)
        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs

class ConvTransposeNd(_ConvTransposeMixin,_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,out_padding,
                 weight_init=Kaiming_Normal(), bias_init=Zeros(), activation=None):
        super(ConvTransposeNd, self).__init__(in_channels, out_channels, kernel_size, stride, padding,
                                     dilation, True, out_padding, groups, bias)
        self.activation = activation

        if weight_init is not None:
            weight_init(self.weight.data)
        if bias and bias_init is not None:
            bias_init(self.bias.data)

    def __call__(self, *args, **kwargs):

        outputs = super(ConvTransposeNd,self).__call__(*args, **kwargs)
        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs

class Conv2d(ConvNd):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,groups=1,bias=True,weight_init=Kaiming_Normal(),bias_init=Zeros(),activation=None):
        super(Conv2d,self).__init__(in_channels,out_channels,_pair(kernel_size),_pair(stride),_pair(padding),_pair(dilation),groups,bias,_pair(0),weight_init,bias_init,activation)


    def forward(self,input):
        return F.conv2d(input,self.weight,self.bias,self.stride,self.padding,self.dilation,self.groups)


class ConvTranspose2d(ConvTransposeNd):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,groups=1,bias=True,weight_init=Kaiming_Normal(),bias_init=Zeros(),activation=None):
        super(ConvTranspose2d,self).__init__(in_channels,out_channels,_pair(kernel_size),_pair(stride),_pair(padding),_pair(dilation),groups,bias,_pair(0),weight_init,bias_init,activation)


    def forward(self,input,output_size=None):
        output_padding = self._output_padding(input, output_size)
        return F.conv_transpose2d(input,self.weight,self.bias,self.stride,self.padding,output_padding,self.groups,self.dilation)



class Conv1d(ConvNd):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,groups=1,bias=True,weight_init=Kaiming_Normal(),bias_init=Zeros(),activation=None):
        super(Conv1d,self).__init__(in_channels,out_channels,_single(kernel_size),_single(stride),_single(padding),_single(dilation),groups,bias,_single(0),weight_init,bias_init,activation)


    def forward(self,input):

        return F.conv1d(input,self.weight,self.bias,self.stride,self.padding,self.dilation,self.groups)


class ConvTranspose1d(ConvTransposeNd):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,groups=1,bias=True,weight_init=Kaiming_Normal(),bias_init=Zeros(),activation=None):
        super(ConvTranspose1d,self).__init__(in_channels,out_channels,_single(kernel_size),_single(stride),_single(padding),_single(dilation),groups,bias,_single(0),weight_init,bias_init,activation)


    def forward(self,input,output_size=None):
        output_padding = self._output_padding(input, output_size)
        return F.conv_transpose1d(input,self.weight,self.bias,self.stride,self.padding,output_padding,self.dilation,self.groups)


class Conv3d(ConvNd):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,groups=1,bias=True,weight_init=Kaiming_Normal(),bias_init=Zeros(),activation=None):
        super(Conv3d,self).__init__(in_channels,out_channels,_triple(kernel_size),_triple(stride),_triple(padding),_triple(dilation),groups,bias,_triple(0),weight_init,bias_init,activation)


    def forward(self,input):
        return F.conv3d(input,self.weight,self.bias,self.stride,self.padding,self.dilation,self.groups)


class ConvTranspose3d(ConvTransposeNd):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,groups=1,bias=True,weight_init=Kaiming_Normal(),bias_init=Zeros(),activation=None):
        super(ConvTranspose3d,self).__init__(in_channels,out_channels,_triple(kernel_size),_triple(stride),_triple(padding),_triple(dilation),groups,bias,_triple(0),weight_init,bias_init,activation)


    def forward(self,input,output_size=None):
        output_padding = self._output_padding(input, output_size)
        return F.conv_transpose3d(input,self.weight,self.bias,self.stride,self.padding,output_padding,self.dilation,self.groups)



class DepthwiseConv2d(ConvNd):
    def __init__(self,in_channels,kernel_size,stride=1,padding=0,dilation=1,bias=True,weight_init=Kaiming_Normal(),bias_init=Zeros(),activation=None):
        super(DepthwiseConv2d,self).__init__(in_channels,in_channels,_pair(kernel_size),_pair(stride),_pair(padding),_pair(dilation),in_channels,bias,_pair(0),weight_init,bias_init,activation)


    def forward(self,input):
        return F.conv2d(input,self.weight,self.bias,self.stride,self.padding,self.dilation,self.groups)

class DepthwiseConvTranspose2d(ConvTransposeNd):
    def __init__(self,in_channels,kernel_size,stride=1,padding=0,dilation=1,bias=True,weight_init=Kaiming_Normal(),bias_init=Zeros(),activation=None):
        super(DepthwiseConvTranspose2d,self).__init__(in_channels,in_channels,_pair(kernel_size),_pair(stride),_pair(padding),_pair(dilation),in_channels,bias,_pair(0),weight_init,bias_init,activation)


    def forward(self,input,output_size=None):
        output_padding = self._output_padding(input, output_size)
        return F.conv_transpose2d(input,self.weight,self.bias,self.stride,self.padding,output_padding,self.dilation,self.groups)

class DepthwiseConv1d(ConvNd):
    def __init__(self,in_channels,kernel_size,stride=1,padding=0,dilation=1,bias=True,weight_init=Kaiming_Normal(),bias_init=Zeros(),activation=None):
        super(DepthwiseConv1d,self).__init__(in_channels,in_channels,_single(kernel_size),_single(stride),_single(padding),_single(dilation),in_channels,bias,_single(0),weight_init,bias_init,activation)


    def forward(self,input):
        return F.conv1d(input,self.weight,self.bias,self.stride,self.padding,self.dilation,self.groups)

class DepthwiseConvTranspose1d(ConvTransposeNd):
    def __init__(self,in_channels,kernel_size,stride=1,padding=0,dilation=1,bias=True,weight_init=Kaiming_Normal(),bias_init=Zeros(),activation=None):
        super(DepthwiseConvTranspose1d,self).__init__(in_channels,in_channels,_single(kernel_size),_single(stride),_single(padding),_single(dilation),in_channels,bias,_single(0),weight_init,bias_init,activation)


    def forward(self,input,output_size=None):
        output_padding = self._output_padding(input, output_size)
        return F.conv_transpose1d(input,self.weight,self.bias,self.stride,self.padding,output_padding,self.dilation,self.groups)

class DepthwiseConv3d(ConvNd):
    def __init__(self,in_channels,kernel_size,stride=1,padding=0,dilation=1,bias=True,weight_init=Kaiming_Normal(),bias_init=Zeros(),activation=None):
        super(DepthwiseConv3d,self).__init__(in_channels,in_channels,_triple(kernel_size),_triple(stride),_triple(padding),_triple(dilation),in_channels,bias,_triple(0),weight_init,bias_init,activation)


    def forward(self,input):
        return F.conv3d(input,self.weight,self.bias,self.stride,self.padding,self.dilation,self.groups)

class DepthwiseConvTranspose3d(ConvTransposeNd):
    def __init__(self,in_channels,kernel_size,stride=1,padding=0,dilation=1,bias=True,weight_init=Kaiming_Normal(),bias_init=Zeros(),activation=None):
        super(DepthwiseConvTranspose3d,self).__init__(in_channels,in_channels,_triple(kernel_size),_triple(stride),_triple(padding),_triple(dilation),in_channels,bias,_triple(0),weight_init,bias_init,activation)


    def forward(self,input,output_size=None):
        output_padding = self._output_padding(input, output_size)
        return F.conv_transpose3d(input,self.weight,self.bias,self.stride,self.padding,output_padding,self.dilation,self.groups)

class Linear(nn.Linear):
    def __init__(self,in_features,out_features,bias=True,weight_init=Xavier_Normal(),bias_init=Zeros(),activation=None):
        super(Linear,self).__init__(in_features,out_features,bias)
        self.activation = activation

        if weight_init is not None:
            weight_init(self.weight.data)
        if bias and bias_init is not None:
            bias_init(self.bias.data)

    def forward(self,input):
        output = F.linear(input,self.weight,self.bias)
        return output

    def __call__(self, *args, **kwargs):
        outputs = super().__call__(*args,**kwargs)
        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs

class Flatten(nn.Module):
    def __init__(self,batch_first=True):
        super(Flatten,self).__init__()
        self.batch_first = batch_first

    def forward(self,inputs):

        if self.batch_first:
            size = torch.prod(torch.LongTensor(list(inputs.size())[1:])).item()
            return inputs.view(-1,size)
        else:
            size = torch.prod(torch.LongTensor(list(inputs.size())[:len(inputs.size())-1])).item()
            return inputs.view(size,-1)

class Reshape(nn.Module):
    def __init__(self,output_shape,batch_first=True):
        super(Reshape,self).__init__()

        self.output_shape = output_shape
        self.batch_first = batch_first

    def forward(self,inputs):
        if isinstance(self.output_shape,int):
            size = [self.output_shape]
        else:
            size = list(self.output_shape)

        if self.batch_first:
            input_total_size = torch.prod(torch.LongTensor(list(inputs.size())[1:])).item()
        else:
            input_total_size = torch.prod(torch.LongTensor(list(inputs.size())[:len(inputs.size())-1])).item()



        target_total_size = torch.prod(torch.LongTensor(size)).item()

        if input_total_size != target_total_size:
            raise ValueError(" Reshape must preserve total dimension, input size: {} and output size: {}".format(input.size()[1:],self.output_shape))

        size = list(size)
        if self.batch_first:
            size = tuple([-1] + size)
        else:
            size = tuple(size + [-1])
        outputs = inputs.view(size)

        return outputs


class _GlobalPoolNd(nn.Module):
    def __init__(self,flatten=True):
        super(_GlobalPoolNd,self).__init__()
        self.flatten = flatten

    def pool(self,input):
        raise NotImplementedError()

    def forward(self,input):
        input = self.pool(input)
        size_0 = input.size(1)
        return input.view(-1,size_0) if self.flatten else input

class GlobalAvgPool1d(_GlobalPoolNd):
    def __init__(self,flatten=True):
        super().__init__(flatten)

    def pool(self, input):
        return F.adaptive_avg_pool1d(input,1)

class GlobalAvgPool2d(_GlobalPoolNd):
    def __init__(self, flatten=True):
        super().__init__(flatten)

    def pool(self, input):
        return F.adaptive_avg_pool2d(input,1)

class GlobalAvgPool3d(_GlobalPoolNd):
    def __init__(self, flatten=True):
        super().__init__(flatten)

    def pool(self, input):
        return F.adaptive_avg_pool3d(input,1)


class GlobalMaxPool1d(_GlobalPoolNd):
    def __init__(self, flatten=True):
        super().__init__(flatten)

    def pool(self, input):
        return F.adaptive_max_pool1d(input, 1)


class GlobalMaxPool2d(_GlobalPoolNd):
    def __init__(self, flatten=True):
        super().__init__(flatten)

    def pool(self, input):
        return F.adaptive_max_pool2d(input, 1)


class GlobalMaxPool3d(_GlobalPoolNd):
    def __init__(self, flatten=True):
        super().__init__(flatten)

    def pool(self, input):
        return F.adaptive_max_pool3d(input, 1)


class RNNBase(nn.RNNBase):
    def __init__(self,mode, input_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0, bidirectional=False,weight_init=None):
        super(RNNBase,self).__init__(mode, input_size, hidden_size,
                 num_layers, bias, batch_first, dropout,bidirectional)

        if weight_init is not None:
            for weight in super(RNNBase, self).parameters():
                weight_init(weight)

class RNN(RNNBase):
    def __init__(self, *args, **kwargs):
        if 'nonlinearity' in kwargs:
            if kwargs['nonlinearity'] == 'tanh':
                mode = 'RNN_TANH'
            elif kwargs['nonlinearity'] == 'relu':
                mode = 'RNN_RELU'
            else:
                raise ValueError("Unknown nonlinearity '{}'".format(
                    kwargs['nonlinearity']))
            del kwargs['nonlinearity']
        else:
            mode = 'RNN_TANH'

        super(RNN, self).__init__(mode, *args, **kwargs)

class GRU(RNNBase):
    def __init__(self, *args, **kwargs):
            super(GRU, self).__init__('GRU', *args, **kwargs)

class LSTM(RNNBase):
    def __init__(self, *args, **kwargs):
        super(LSTM, self).__init__('LSTM', *args, **kwargs)

#ADD LSTM,norm etc ETC
class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, inputs):

        return inputs * torch.sigmoid(inputs)

class GroupNorm(nn.GroupNorm):
    def __init__(self, *args,weight_init=None,bias_init=None):
        super().__init__(*args)

        if weight_init is not None:
            weight_init(self.weight.data)
        if bias_init is not None:
            bias_init(self.bias.data)

class LayerNorm(nn.LayerNorm):
    def __init__(self, *args,weight_init=None,bias_init=None):
        super().__init__(*args)

        if weight_init is not None:
            weight_init(self.weight.data)
        if bias_init is not None:
            bias_init(self.bias.data)


class Embedding(nn.Embedding):
    def __init__(self,num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2, scale_grad_by_freq=False,
                 sparse=False, _weight=None,weight_init=None):
        super().__init__(num_embeddings, embedding_dim, padding_idx,
                 max_norm, norm_type, scale_grad_by_freq,
                 sparse, _weight)

        if weight_init is not None:
            weight_init(self.weight.data)


class BatchNorm(_BatchNorm):
    def __init__(self,num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True,weight_init=None,bias_init=None):
        super().__init__(num_features, eps, momentum,
                 track_running_stats)

        if weight_init is not None:
            weight_init(self.weight.data)

        if bias_init is not None:
            bias_init(self.bias.data)



class BatchNorm1d(BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))


class BatchNorm2d(BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

class BatchNorm3d(BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)')






