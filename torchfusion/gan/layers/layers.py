from torchfusion.layers import *
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from math import floor,sqrt
from torchfusion.initializers import *
import torch

class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_class, eps=1e-5, momentum=0.1,
                 track_running_stats=True):
        """

        :param num_features:
        :param num_class:
        :param eps:
        :param momentum:
        :param track_running_stats:
        """
        super(ConditionalBatchNorm2d,self).__init__()

        self.bn = BatchNorm2d(num_features=num_features,eps=eps,momentum=momentum,track_running_stats=track_running_stats, affine=False)
        self.gamma_embed = Embedding(num_class, num_features)
        self.beta_embed = Embedding(num_class, num_features)
        self.gamma_embed.weight.data = torch.ones(self.gamma_embed.weight.size())
        self.beta_embed.weight.data = torch.zeros(self.gamma_embed.weight.size())

    def forward(self, input, class_id):
        input = input.float()
        class_id = class_id.long()
        out = self.bn(input)
        gamma = self.gamma_embed(class_id).squeeze(1).unsqueeze(2).unsqueeze(3)
        beta = self.beta_embed(class_id).squeeze(1).unsqueeze(2).unsqueeze(3)

        out = gamma * out.type(gamma.dtype) + beta


        return out


class SelfAttention(nn.Module):
    def __init__(self,in_channels,weight_init=Kaiming_Normal(),bias_init=Zeros(),use_bias=False):
        """

        :param in_channels:
        :param weight_init:
        :param bias_init:
        :param use_bias:
        """
        super(SelfAttention,self).__init__()

        self.q = Conv2d(in_channels,in_channels//8,kernel_size=1,weight_init=weight_init,bias_init=bias_init,bias=use_bias)
        self.k = Conv2d(in_channels,in_channels//8,kernel_size=1,weight_init=weight_init,bias_init=bias_init,bias=use_bias)

        self.v = Conv2d(in_channels,in_channels,kernel_size=1,weight_init=weight_init,bias_init=bias_init,bias=use_bias)

        self.softmax = nn.Softmax(dim=-1)

        self.atten_weight = nn.Parameter(torch.tensor([0.0]))

    def forward(self,input):
        batch_size, channels, width, height = input.size()
        res = input

        queries = self.q(input).view(batch_size,-1,width*height).permute(0,2,1)
        keys = self.k(input).view(batch_size,-1,width*height)
        values = self.v(input).view(batch_size, -1, width * height)

        atten_ = self.softmax(torch.bmm(queries, keys)).permute(0,2,1)

        atten_values = torch.bmm(values,atten_).view(batch_size,channels,width,height)


        return (self.atten_weight * atten_values) + res



class GeneratorResBlock(nn.Module):
    def __init__(self,in_channels,out_channels,num_classes=0,upsample_size=1,kernel_size=3,activation=nn.ReLU(),conv_groups=1,dropout_ratio=0):
        """

        :param in_channels:
        :param out_channels:
        :param num_classes:
        :param upsample_size:
        :param kernel_size:
        :param activation:
        :param conv_groups:
        :param dropout_ratio:
        """
        super(GeneratorResBlock,self).__init__()
        padding = floor(kernel_size/2)
        self.activation = activation
        self.num_classes = num_classes
        self.upsample_size = upsample_size

        self.dropout = nn.Dropout(dropout_ratio)

        self.conv1 = spectral_norm(Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=padding,weight_init=Xavier_Uniform(sqrt(2)),groups=conv_groups))
        self.conv2 = spectral_norm(Conv2d(out_channels,out_channels,kernel_size=kernel_size,padding=padding,weight_init=Xavier_Uniform(sqrt(2)),groups=conv_groups))
        if num_classes > 0:
            self.bn1 = ConditionalBatchNorm2d(in_channels,num_classes)
            self.bn2 = ConditionalBatchNorm2d(out_channels,num_classes)
        else:
            self.bn1 = BatchNorm2d(in_channels)
            self.bn2 = BatchNorm2d(out_channels)

        self.res_upsample = nn.Sequential()

        if in_channels != out_channels or upsample_size > 1:
            self.res_upsample = Conv2d(in_channels, out_channels, kernel_size=1,weight_init=Xavier_Uniform())

    def forward(self,inputs,labels=None):
        res = inputs

        if labels is not None:
            inputs = self.bn1(inputs,labels)
        else:
            inputs = self.bn1(inputs)

        inputs = self.dropout(self.conv1(self.activation(inputs)))
        if self.upsample_size > 1:
            inputs = F.interpolate(inputs,scale_factor=self.upsample_size)

        if labels is not None:
            inputs = self.bn2(inputs,labels)
        else:
            inputs = self.bn2(inputs)

        inputs = self.conv2(self.activation(inputs))

        if self.upsample_size > 1:
            return inputs + F.interpolate(self.res_upsample(res),scale_factor=self.upsample_size)
        else:
            return inputs + self.res_upsample(res)

class StandardGeneratorBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,padding,stride,num_classes=0,activation=nn.LeakyReLU(0.2),conv_groups=1):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param padding:
        :param stride:
        :param num_classes:
        :param activation:
        :param conv_groups:
        """

        super(StandardGeneratorBlock,self).__init__()

        self.activation = activation
        self.num_classes = num_classes


        self.conv = spectral_norm(ConvTranspose2d(in_channels,out_channels,kernel_size=kernel_size,padding=padding,stride=stride,weight_init=Xavier_Uniform(),groups=conv_groups))
        if num_classes > 0:
            self.bn = ConditionalBatchNorm2d(out_channels,num_classes)
        else:
            self.bn = BatchNorm2d(out_channels)

    def forward(self,inputs,labels=None):

        inputs = self.conv(inputs)

        if labels is not None:
            inputs = self.bn(inputs,labels)
        else:
            inputs = self.bn(inputs)

        inputs = self.activation(inputs)

        return inputs



class DiscriminatorResBlock(nn.Module):
    def __init__(self,in_channels,out_channels,downsample_size=1,kernel_size=3,activation=nn.ReLU(),initial_activation=True,conv_groups=1,dropout_ratio=0):
        """

        :param in_channels:
        :param out_channels:
        :param downsample_size:
        :param kernel_size:
        :param activation:
        :param initial_activation:
        :param conv_groups:
        """

        super(DiscriminatorResBlock,self).__init__()

        padding = floor(kernel_size / 2)
        self.activation = activation
        self.initial_activation = initial_activation
        self.dropout = nn.Dropout(dropout_ratio)

        self.conv1 = spectral_norm(Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=padding,weight_init=Xavier_Uniform(),groups=conv_groups))
        self.conv2 = spectral_norm(Conv2d(out_channels,out_channels,kernel_size=kernel_size,padding=padding,weight_init=Xavier_Uniform(),groups=conv_groups))
        self.downsample = nn.Sequential()
        if downsample_size > 1:
            self.downsample = nn.AvgPool2d(kernel_size=downsample_size)

        self.res_downsample = nn.Sequential()
        if in_channels != out_channels or downsample_size  > 1:
            self.res_downsample = nn.Sequential(
                Conv2d(in_channels,out_channels,kernel_size=1,weight_init=Xavier_Uniform(sqrt(2))),
                nn.AvgPool2d(kernel_size=downsample_size)
            )

    def forward(self,inputs):

        res = inputs

        if self.initial_activation:
            inputs = self.activation(inputs)

        inputs = self.conv1(inputs)
        inputs = self.dropout(self.activation(inputs))
        inputs = self.conv2(inputs)
        inputs = self.downsample(inputs)

        return inputs + self.res_downsample(res)




class StandardDiscriminatorBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,padding,stride,activation=nn.LeakyReLU(0.2),use_bn=False,conv_groups=1):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param padding:
        :param stride:
        :param activation:
        :param use_bn:
        :param conv_groups:
        """

        super(StandardDiscriminatorBlock,self).__init__()

        self.activation = activation

        self.conv = spectral_norm(Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=padding,stride=stride,weight_init=Xavier_Uniform(),groups=conv_groups))

        self.bn = nn.Sequential()

        if use_bn:
            self.bn = BatchNorm2d(out_channels,weight_init=Normal(1.0,0.02))

    def forward(self,inputs):

        return self.activation(self.bn(self.conv(inputs)))

