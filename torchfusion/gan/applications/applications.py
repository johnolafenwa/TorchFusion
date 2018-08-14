from .layers import *
from torchfusion.initializers import *
from math import floor


class ResGenerator(nn.Module):
    def __init__(self,output_size,num_classes=0,latent_size=100,kernel_size=3,activation=nn.ReLU(),conv_groups=1,attention=False):
        super().__init__()

        padding = floor(kernel_size/2)

        self.num_classes = num_classes

        output_channels = output_size[0]
        self.size = output_size[1]

        self.fc = Linear(latent_size,4 * 4 * self.size * 8)

        current_size = 4

        self.layers = nn.ModuleList()

        in_channels = self.size * 8



        while current_size < self.size:

            self.layers.append(GeneratorResBlock(in_channels,in_channels // 2,num_classes,upsample_size=2,kernel_size=kernel_size,activation=activation,conv_groups=conv_groups if current_size == 4 else 1))
            current_size *= 2

            in_channels = in_channels//2

            if current_size == self.size // 2 and attention:
                self.layers.append(SelfAttention(in_channels))


        self.net = nn.Sequential(
            BatchNorm2d(in_channels, weight_init=Normal(1.0, 0.02)),
            Conv2d(in_channels, output_channels, kernel_size=kernel_size, padding=padding, weight_init=Xavier_Uniform()),
            nn.Tanh()
        )

    def forward(self,inputs,labels=None):
        outputs = self.fc(inputs).view(-1,self.size * 8,4,4)

        for layer in self.layers:
            if self.num_classes > 1:
                outputs = layer(outputs,labels)
            else:
                outputs = layer(outputs)


        return self.net(outputs)


class StandardGenerator(nn.Module):
    def __init__(self,output_size,num_classes=0,latent_size=100,activation=nn.LeakyReLU(0.2),conv_groups=1,attention=False):
        super().__init__()

        output_channels = output_size[0]
        self.size = output_size[1]
        self.latent_size = latent_size
        self.num_classes = num_classes

        current_size = 4

        self.layers = nn.ModuleList()

        in_channels = self.size * 8

        self.layers.append(StandardGeneratorBlock(latent_size,in_channels,num_classes=num_classes,kernel_size=4,padding=0,stride=1,activation=activation))

        while current_size < self.size:
            current_size *= 2

            if current_size < self.size:
                self.layers.append(StandardGeneratorBlock(in_channels,in_channels // 2,num_classes=num_classes,kernel_size=4,stride=2,padding=1,activation=activation,conv_groups=conv_groups))
                in_channels = in_channels // 2

                if current_size == self.size // 2 and attention:
                    self.layers.append(SelfAttention(in_channels))
        self.final_conv = spectral_norm(ConvTranspose2d(in_channels,output_channels,kernel_size=4,stride=2,padding=1,weight_init=Xavier_Uniform()))

    def forward(self,inputs,labels=None):
        outputs = inputs.view(-1,self.latent_size ,1,1)

        for layer in self.layers:
            if self.num_classes > 1:
                outputs = layer(outputs,labels)
            else:
                outputs = layer(outputs)


        return torch.tanh(self.final_conv(outputs))


class ResProjectionDiscriminator(nn.Module):
    def __init__(self,input_size,num_classes=0,kernel_size=3,activation=nn.ReLU(),attention=True,apply_sigmoid=False,conv_groups=1):
        super().__init__()
        self.num_classes = num_classes
        in_channels = input_size[0]
        out_channels = in_channels
        size = input_size[1]
        self.apply_sigmoid = apply_sigmoid

        layers = [DiscriminatorResBlock(in_channels,size,kernel_size=kernel_size,activation=activation,initial_activation=False)]

        current_size = size
        in_channels = size

        while current_size > 4:
            layers.append(DiscriminatorResBlock(in_channels,in_channels * 2,kernel_size=kernel_size,downsample_size=2,activation=activation,conv_groups=conv_groups))
            current_size /= 2
            in_channels *= 2
            if current_size == size//2 and attention:
                layers.append(SelfAttention(in_channels))

        layers.append(GlobalAvgPool2d())

        self.fc = spectral_norm(Linear(in_channels,out_channels,weight_init=Xavier_Uniform()))

        self.net = nn.Sequential(*layers)

        if self.num_classes > 1:

            self.embed = spectral_norm(Embedding(num_classes,in_channels,weight_init=Xavier_Uniform()))

    def forward(self,inputs,labels=None):
        outputs = self.net(inputs)

        linear_out = self.fc(outputs)

        if self.num_classes > 1:
            embed = self.embed(labels.long()).squeeze(1)

            size = outputs.size(1)

            dot = torch.bmm(outputs.view(-1,1,size),embed.view(-1,size,1)).squeeze(2)

            return torch.sigmoid(linear_out + dot) if self.apply_sigmoid else linear_out + dot
        else:
            return torch.sigmoid(linear_out) if self.apply_sigmoid else linear_out

class StandardProjectionDiscriminator(nn.Module):
    def __init__(self,input_size,num_classes=0,activation=nn.LeakyReLU(0.2),attention=True,apply_sigmoid=True,use_bn=False,conv_groups=1):
        super().__init__()
        self.num_classes = num_classes
        in_channels = input_size[0]
        out_channels = in_channels
        size = input_size[1]
        self.apply_sigmoid = apply_sigmoid

        layers = [StandardDiscriminatorBlock(in_channels,size,kernel_size=3,stride=1,padding=1,use_bn=use_bn,activation=activation)]

        current_size = size
        in_channels = size

        while current_size > 4:
            layers.append(StandardDiscriminatorBlock(in_channels,in_channels * 2,kernel_size=4,stride=2,padding=1,use_bn=use_bn,conv_groups=conv_groups))
            current_size /= 2
            in_channels *= 2
            if current_size == size//2 and attention:
                layers.append(SelfAttention(in_channels))

        layers.append(Flatten())
        self.fc = spectral_norm(Linear(in_channels * 16,1,weight_init=Xavier_Uniform()))

        self.net = nn.Sequential(*layers)

        if self.num_classes > 1:

            self.embed = spectral_norm(Embedding(num_classes,in_channels * 16,weight_init=Xavier_Uniform()))

    def forward(self,inputs,labels=None):
        outputs = self.net(inputs)

        linear_out = self.fc(outputs)

        if self.num_classes > 1:
            embed = self.embed(labels.long()).squeeze(1)

            size = outputs.size(1)

            dot = torch.bmm(outputs.view(-1,1,size),embed.view(-1,size,1)).squeeze(2)

            return torch.sigmoid(linear_out + dot) if self.apply_sigmoid else linear_out + dot
        else:
            return torch.sigmoid(linear_out) if self.apply_sigmoid else linear_out

class DCGANGenerator(nn.Module):
    def __init__(self,latent_size,output_size,dropout_ratio=0.0,use_bias=False,num_gpus=1):
        super(DCGANGenerator,self).__init__()

        assert output_size[1] >= 32

        self.num_gpus = num_gpus

        in_channels = latent_size[0]

        multiplier = 8
        out_size = output_size[1]
        layers = [ConvTranspose2d(in_channels=in_channels,out_channels=int(out_size * multiplier),kernel_size=4,stride=1,padding=0,bias=use_bias,weight_init=Normal(0.0,0.02)),
                  BatchNorm2d(int(out_size * multiplier),weight_init=Normal(1.0,0.02)),
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
                layers.append(ConvTranspose2d(in_channels=in_channels, out_channels=output_size[0], kernel_size=4, stride=2, padding=1, bias=use_bias,weight_init=Normal(0.0,0.02)))
                layers.append(nn.Tanh())
            else:
                layers.append(ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=use_bias,weight_init=Normal(0.0,0.02)))
                layers.append(BatchNorm2d(out_channels,weight_init=Normal(1.0,0.02)))
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

class DCGANDiscriminator(nn.Module):
    def __init__(self,input_size,dropout_ratio=0.0,use_bias=False,num_gpus=1,apply_sigmoid=True):
        super(DCGANDiscriminator,self).__init__()

        assert input_size[1] >= 32

        self.num_gpus = num_gpus

        input_channels = input_size[0]
        in_channels = input_channels
        size = input_size[1]
        self.apply_sigmoid = apply_sigmoid

        channel_multiplier = 1

        out_channels = size

        layers = []

        while size > 4:
            layers.append(Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=4,stride=2,padding=1,bias=use_bias,weight_init=Normal(0.0,0.02)))
            if size != input_size[1]:
                layers.append(BatchNorm2d(out_channels,weight_init=Normal(1.0,0.02)))
            layers.append(nn.LeakyReLU(0.2,inplace=True))
            layers.append(nn.Dropout(dropout_ratio))
            if channel_multiplier < 8:
                channel_multiplier *= 2
            size /= 2


            in_channels = out_channels
            out_channels = input_size[1] * channel_multiplier

        layers.append(Conv2d(in_channels=in_channels,out_channels=1,kernel_size=4,padding=0,bias=use_bias,weight_init=Normal(0.0,0.02)))

        self.net = nn.Sequential(*layers)


    def forward(self,input):

        if input.is_cuda and self.num_gpus > 1:
            output = nn.parallel.data_parallel(self.net,input,range(self.num_gpus))
        else:
            output = self.net(input)

        return torch.sigmoid(output.view(-1,1)) if self.apply_sigmoid else output.view(-1,1)

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
            layers.append(Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=4,stride=2,padding=1,bias=use_bias,weight_init=Normal(0.0,0.02)))
            layers.append(nn.LeakyReLU(0.2,inplace=True))
            layers.append(nn.Dropout(dropout_ratio))
            if channel_multiplier < 8:
                channel_multiplier *= 2
            size /= 2


            in_channels = out_channels
            out_channels = input_size[1] * channel_multiplier

        layers.append(Conv2d(in_channels=in_channels,out_channels=1,kernel_size=4,padding=0,bias=use_bias,weight_init=Normal(0.0,0.02)))

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
        layers.append(Linear(latent_size, hidden_dims))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Dropout(dropout_ratio))

        for i in range(depth - 2):
            layers.append(Linear(hidden_dims, hidden_dims))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(dropout_ratio))

        layers.append(Linear(hidden_dims,output_size[0]*output_size[1] * output_size[2]))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self,input):
        if input.is_cuda and self.num_gpus > 1:
            output = nn.parallel.data_parallel(self.net,input,range(self.num_gpus))
        else:
            output = self.net(input)

        return output.view(-1,self.output_size[0],self.output_size[1] ,self.output_size[2])

class MLPDiscriminator(nn.Module):
    def __init__(self,input_size,hidden_dims=512,depth=4,dropout_ratio=0.0,num_gpus=1,apply_sigmoid=True):
        super(MLPDiscriminator,self).__init__()
        self.num_gpus = num_gpus
        self.input_size = input_size
        self.apply_sigmoid = apply_sigmoid

        layers = []
        layers.append(Linear(input_size[0] * input_size[1] * input_size[2], hidden_dims))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Dropout(dropout_ratio))

        for i in range(depth - 2):
            layers.append(Linear(hidden_dims, hidden_dims))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(dropout_ratio))

        layers.append(Linear(hidden_dims, 1))

        self.net = nn.Sequential(*layers)
    def forward(self,input):
        input = input.view(-1,input.size(1) * input.size(2) * input.size(3))
        if input.is_cuda and self.num_gpus > 1:
            output = nn.parallel.data_parallel(self.net,input,range(self.num_gpus))
        else:
            output = self.net(input)

        return torch.sigmoid(output.view(-1, 1)) if self.apply_sigmoid else output.view(-1,1)

class WMLPDiscriminator(nn.Module):
    def __init__(self,input_size,hidden_dims=512,depth=4,dropout_ratio=0.0,num_gpus=1):
        super(WMLPDiscriminator,self).__init__()
        self.num_gpus = num_gpus
        self.input_size = input_size

        layers = []
        layers.append(Linear(input_size[0] * input_size[1] * input_size[2],hidden_dims))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Dropout(dropout_ratio))

        for i in range(depth - 2):
            layers.append(Linear(hidden_dims,hidden_dims))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(dropout_ratio))

        layers.append(Linear(hidden_dims,1))

        self.net = nn.Sequential(*layers)

    def forward(self,input):
        input = input.view(-1,input.size(1) * input.size(2) * input.size(3))
        if input.is_cuda and self.num_gpus > 1:
            output = nn.parallel.data_parallel(self.net,input,range(self.num_gpus))
        else:
            output = self.net(input)

        return output.view(-1, 1)