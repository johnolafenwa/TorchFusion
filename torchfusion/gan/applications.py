import torch.nn as nn


""" The standard DCGAN Generator as proposed by Radford et al. 2015 (https://arxiv.org/1511.06434)
    latent_size: the size of the noise vector
    output_size: the size of the image to be generated
    dropout_ratio: Dropout rate for applying dropout after every Relu layer
    use_bias: Enables or disables bias in the convolution layers
    num_gpus: Parallelizes computation over the number of GPUs specified.

"""
class DCGANGenerator(nn.Module):
    def __init__(self,latent_size,output_size,dropout_ratio=0.0,use_bias=False,num_gpus=1):
        super(DCGANGenerator,self).__init__()

        assert output_size[1] >= 32

        self.num_gpus = num_gpus

        in_channels = latent_size[0]

        multiplier = 8
        out_size = output_size[1]
        layers = [nn.ConvTranspose2d(in_channels=in_channels,out_channels=int(out_size * multiplier),kernel_size=4,stride=1,padding=0,bias=use_bias),
                  nn.BatchNorm2d(int(out_size * multiplier)),
                  nn.ReLU(inplace=True),
                  nn.Dropout(dropout_ratio)
                  ]

        in_channels = int(out_size * multiplier)

        size = 4 * latent_size[1]
        while size < output_size[1]:
            multiplier /= 2
            size *= 2

            if size < int(out_size * multiplier):
                out_channels = int(out_size * multiplier)
            else:
                out_channels = out_size
            if size == output_size[1]:
                layers.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=output_size[0], kernel_size=4, stride=2, padding=1, bias=use_bias))
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=use_bias))
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout_ratio))
                in_channels = out_channels

        self.net = nn.Sequential(*layers)

    def forward(self,inputs):

        if inputs.is_cuda and self.num_gpus > 1:
            out = nn.parallel.data_parallel(self.net,inputs,range(self.num_gpus))
        else:
            out = self.net(inputs)

        return out


""" The standard DCGAN Discriminator as proposed by Radford et al. 2015 (https://arxiv.org/1511.06434)
    input_size: the size of the input image
    dropout_ratio: Dropout rate for applying dropout after every Relu layer
    use_bias: Enables or disables bias in the convolution layers
    num_gpus: Parallelizes computation over the number of GPUs specified.

"""
class DCGANDiscriminator(nn.Module):
    def __init__(self,input_size,dropout_ratio=0.0,use_bias=False,num_gpus=1):
        super(DCGANDiscriminator,self).__init__()

        assert input_size[1] >= 32

        self.num_gpus = num_gpus

        input_channels = input_size[0]
        in_channels = input_channels
        size = input_size[1]

        channel_multiplier = 1

        out_channels = size

        layers = []

        while size > 4:
            layers.append(nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=4,stride=2,padding=1,bias=use_bias))
            if size != input_size[1]:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2,inplace=True))
            layers.append(nn.Dropout(dropout_ratio))
            if channel_multiplier < 8:
                channel_multiplier *= 2
            size /= 2


            in_channels = out_channels
            out_channels = input_size[1] * channel_multiplier

        layers.append(nn.Conv2d(in_channels=in_channels,out_channels=1,kernel_size=4,padding=0,bias=use_bias))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)


    def forward(self,input):

        if input.is_cuda and self.num_gpus > 1:
            output = nn.parallel.data_parallel(self.net,input,range(self.num_gpus))
        else:
            output = self.net(input)

        return output.view(-1,1).squeeze(1)


""" Wasserstein DCGAN Discriminator as proposed by Gulrajani et al. 2017 (https://arxiv.org/1704.00028)
    based on earlier work by Arjovsky et al. 2017 (https://arxiv.org/1701.07875)
   
    input_size: the size of the input image
    dropout_ratio: Dropout rate for applying dropout after every Relu layer
    use_bias: Enables or disables bias in the convolution layers
    num_gpus: Parallelizes computation over the number of GPUs specified.

"""
class WGANDiscriminator(nn.Module):
    def __init__(self,input_size,dropout_ratio=0.0,use_bias=False,num_gpus=1):
        super(WGANDiscriminator,self).__init__()

        assert input_size[1] >= 32

        self.num_gpus = num_gpus

        input_channels = input_size[0]
        in_channels = input_channels
        size = input_size[1]

        channel_multiplier = 1

        out_channels = size

        layers = []

        while size > 4:
            layers.append(nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=4,stride=2,padding=1,bias=use_bias))
            layers.append(nn.LeakyReLU(0.2,inplace=True))
            layers.append(nn.Dropout(dropout_ratio))
            if channel_multiplier < 8:
                channel_multiplier *= 2
            size /= 2


            in_channels = out_channels
            out_channels = input_size[1] * channel_multiplier

        layers.append(nn.Conv2d(in_channels=in_channels,out_channels=1,kernel_size=4,padding=0,bias=use_bias))

        self.net = nn.Sequential(*layers)


    def forward(self,input):

        if input.is_cuda and self.num_gpus > 1:
            output = nn.parallel.data_parallel(self.net,input,range(self.num_gpus))
        else:
            output = self.net(input)

        return output.view(-1,1).squeeze(1)

class MLPGenerator(nn.Module):
    def __init__(self,latent_size,output_size,hidden_dims=512,depth=4,dropout_ratio=0.0,num_gpus=1):
        super(MLPGenerator,self).__init__()
        self.num_gpus = num_gpus
        self.output_size = output_size

        layers = []
        layers.append(nn.Linear(latent_size, hidden_dims))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Dropout(dropout_ratio))

        for i in range(depth - 2):
            layers.append(nn.Linear(hidden_dims, hidden_dims))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(dropout_ratio))

        layers.append(nn.Linear(hidden_dims,output_size[0]*output_size[1] * output_size[2]))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self,input):
        if input.is_cuda and self.num_gpus > 1:
            output = nn.parallel.data_parallel(self.net,input,range(self.num_gpus))
        else:
            output = self.net(input)

        return output.view(-1,self.output_size[0],self.output_size[1] ,self.output_size[2])

class MLPDiscriminator(nn.Module):
    def __init__(self,input_size,hidden_dims=512,depth=4,dropout_ratio=0.0,num_gpus=1):
        super(MLPDiscriminator,self).__init__()
        self.num_gpus = num_gpus
        self.input_size = input_size

        layers = []
        layers.append(nn.Linear(input_size[0] * input_size[1] * input_size[2], hidden_dims))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Dropout(dropout_ratio))

        for i in range(depth - 2):
            layers.append(nn.Linear(hidden_dims, hidden_dims))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(dropout_ratio))

        layers.append(nn.Linear(hidden_dims, 1))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)
    def forward(self,input):
        input = input.view(-1,input.size(1) * input.size(2) * input.size(3))
        if input.is_cuda and self.num_gpus > 1:
            output = nn.parallel.data_parallel(self.net,input,range(self.num_gpus))
        else:
            output = self.net(input)

            return output.view(-1, 1).squeeze(1)

class WMLPDiscriminator(nn.Module):
    def __init__(self,input_size,hidden_dims=512,depth=4,dropout_ratio=0.0,num_gpus=1):
        super(WMLPDiscriminator,self).__init__()
        self.num_gpus = num_gpus
        self.input_size = input_size

        layers = []
        layers.append(nn.Linear(input_size[0] * input_size[1] * input_size[2],hidden_dims))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Dropout(dropout_ratio))

        for i in range(depth - 2):
            layers.append(nn.Linear(hidden_dims,hidden_dims))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(dropout_ratio))

        layers.append(nn.Linear(hidden_dims,1))

        self.net = nn.Sequential(*layers)

    def forward(self,input):
        input = input.view(-1,input.size(1) * input.size(2) * input.size(3))
        if input.is_cuda and self.num_gpus > 1:
            output = nn.parallel.data_parallel(self.net,input,range(self.num_gpus))
        else:
            output = self.net(input)

            return output.view(-1, 1).squeeze(1)







