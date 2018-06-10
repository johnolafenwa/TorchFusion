import torch.nn as nn
import torch



class DepthwiseConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, in_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(DepthwiseConvTranspose1d, self).__init__(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                                              padding=padding, dilation=dilation, groups=in_channels, bias=bias)


class DepthwiseConv1d(nn.Conv1d):
    def __init__(self, in_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(DepthwiseConv1d, self).__init__(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                                              padding=padding, dilation=dilation, groups=in_channels, bias=bias)



class DepthwiseConv2d(nn.Conv2d):
    def __init__(self,in_channels,kernel_size,stride=1,padding=0,dilation=1,bias=True):
        super(DepthwiseConv2d,self).__init__(in_channels,in_channels,kernel_size=kernel_size,stride=stride,
                                             padding=padding,dilation=dilation,groups=in_channels,bias=bias)

class DepthwiseConv3d(nn.Conv3d):
    def __init__(self, in_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(DepthwiseConv3d, self).__init__(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                                              padding=padding, dilation=dilation, groups=in_channels, bias=bias)



class DepthwiseConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self,in_channels,kernel_size,stride=1,padding=0,dilation=1,bias=True):
        super(DepthwiseConvTranspose2d,self).__init__(in_channels,in_channels,kernel_size=kernel_size,stride=stride,
                                             padding=padding,dilation=dilation,groups=in_channels,bias=bias)


class DepthwiseConvTranspose3d(nn.ConvTranspose3d):
    def __init__(self, in_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(DepthwiseConvTranspose3d, self).__init__(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                                              padding=padding, dilation=dilation, groups=in_channels, bias=bias)



class Flatten(nn.Module):
    def __init__(self):
        super(Flatten,self).__init__()

    def forward(self,inputs):
        size = torch.prod(torch.LongTensor(list(inputs.size())[1:])).item()

        return inputs.view((-1,size))

class Reshape(nn.Module):
    def __init__(self,output_shape):
        super(Reshape,self).__init__()

        self.output_shape = output_shape

    def forward(self,inputs):
        if isinstance(self.output_shape,int):
            size = [self.output_shape]
        else:
            size = list(self.output_shape)

        assert torch.prod(torch.LongTensor(list(inputs.size())[1:])).item() == torch.prod(torch.LongTensor(size)).item()

        size = list(size)
        size = tuple([-1] + size)
        outputs = inputs.view(size)

        return outputs


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d,self).__init__()


    def forward(self,inputs):
        size0, size1,size2 = inputs.size(1), inputs.size(2),inputs.size(3)

        outputs = nn.AvgPool2d((size1,size2))(inputs).view(-1,size0)

        return outputs

class GlobalMaxPool2d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool2d,self).__init__()


    def forward(self,inputs):
        size0, size1,size2 = inputs.size(1), inputs.size(2),inputs.size(3)

        outputs = nn.MaxPool2d((size1,size2))(inputs).view(-1,size0)

        return outputs
