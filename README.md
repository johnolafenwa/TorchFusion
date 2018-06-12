# TorchFusion 

A modern deep learning framework built to accelerate research and development of AI systems.

Based on PyTorch and fully compatible with pure PyTorch and other pytorch packages, <b>TorchFusion</b> provides a comprehensive extensible training framework
with trainers that you can easily use to train, evaluate and run inference with your PyTorch models, A GAN framework that greatly simplifies the process of
experimenting with Generative Adversarial Networks [Goodfellow et al. 2014](https://arxiv.org/1406.2661), with concrete implementations of a number of GAN algorithms, and a number of high level network layers and utilities to help you be more productive in your work.

<p>The framework is highly extensible, so you can easily create your own custom trainers for specific purposes.</p>

<h2> Unique features of our Trainers</h2>
<ol>
  <li> <h3>Highly configurable </h3></li>
  <li><h3>Highly detailed summary function that not only provides you details about number of parameters, layers, input and output sizes
    but also provides the number of Flops(Multiply-Adds) for every Linear and Convolution layer in your network.
    Now, you can know the exact computational cost of any CNN architecure with just a single function!!!
    </h3>
   <li> <h3>Live metrics and loss visualizations, with option to save them permanently </h3></li>
   <li><h3>Support for persisting logs permanently</h3></li>
  <li> <h3>Easy to use callbacks</h3></li>
</ol>

<b>Note: This is a pre-release version of <b>TorchFusion</b>, the current set of features are just a sneak peek into what is coming!
Future releases of TorchFusion will cut across multiple domains of Deep Learning.
</b>

An <b>AI Commons</b> project <a href="https://commons.specpal.science" >https://commons.specpal.science </a>
Developed and Maintained by [John Olafenwa](https://twitter.com/johnolafenwa) and [Moses Olafenwa](https://twitter.com/OlafenwaMoses), brothers, creators of [ImageAI](https://github.com/OlafenwaMoses/ImageAI )
and Authors of [Introduction to Deep Computer Vision](https://john.specpal.science/deepvision)

<hr>

<h1> Installation </h1>
<h3> Install TorchFusion</h3>
<pre> pip3 install https://github.com/johnolafenwa/TorchFusion/releases/download/0.1.1/torchfusion-0.1.1-py3-none-any.whl </pre>

[Installing PyTorch on Windows](https://pytorch.org)
<h3> CPU Only </h3>
    With Python 3.6 <pre> pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.0-cp36-cp36m-win_amd64.whl torchvision </pre>
    With Python 3.5 <pre> pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.0-cp35-cp35m-win_amd64.whl torchvision </pre>

<h3> With CUDA Support </h3>
    With Python 3.6 <pre> pip3 install http://download.pytorch.org/whl/cu80/torch-0.4.0-cp36-cp36m-win_amd64.whl torchvision </pre>
    With Python 3.5 <pre> pip3 install http://download.pytorch.org/whl/cu80/torch-0.4.0-cp35-cp35m-win_amd64.whl </pre>

[Installing PyTorch on Linux](https://pytorch.org)
<h3> CPU Only </h3>
    With Python 3.6 <pre> pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.0-cp36-cp36m-linux_x86_64.whl  torchvision </pre>
    With Python 3.5 <pre> pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.0-cp35-cp35m-linux_x86_64.whl torchvision </pre>

<h3> With CUDA Support </h3>
    <pre> pip3 install torch  torchvision</pre>


[Installing PyTorch on OSX](https://pytorch.org)
<h3> CPU Only </h3>
    <pre>pip3 install torch  torchvision </pre>

With CUDA Support: Visit [Pytorch.org](https://pytorch.org) for instructions on Installing on OSX with cuda support

<br><br>
<h1>MNIST in Five Minutes</h1>

<pre>
import torchfusion as tf
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torch.nn as nn
from torch.optim import Adam
import torch.cuda as cuda

#Define a the classifier network
net = nn.Sequential(
            tf.Flatten(),
            nn.Linear(784, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
)
batch_size = 64

#Transformations and data augmentation
transformations = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

#Load the training and test sets
train_set = MNIST(root="./data",transform=transformations,download=True)
test_set = MNIST(root="./data",train=False,transform=transformations,download=True)

train_loader = DataLoader(train_set,shuffle=True,batch_size=batch_size,num_workers=4)
test_loader = DataLoader(test_set,shuffle=False,batch_size=batch_size,num_workers=4)

#Move to GPU if available
if cuda.is_available():
    net.cuda()

#Setup the optimize and a loss function
optimizer = Adam(net.parameters(),lr=0.001)
loss_fn = nn.CrossEntropyLoss()

#Top 1 Train accuracy
train_metrics = tf.Accuracy(topK=1)

#Top 1 and Top 2 test accuracy
test_metrics_top1 = tf.Accuracy(name="Top 1 Acc ",topK=1)
test_metrics_top2 = tf.Accuracy(name="Top 2 Acc ",topK=2)

#Create an instance of the StandardModel
model = tf.StandardModel(net)

def train():
    #print a summary of the network
    print(model.summary((1,28,28)))
    model.train(train_loader, loss_fn, optimizer, [train_metrics], test_loader,
                [test_metrics_top1, test_metrics_top2], num_epochs=20,
                model_dir="mnist_mlp_saved_models",save_logs="logs.txt")


if __name__ == "__main__":
    train()


</pre>


<br><br>
<h1>GAN in Five Minutes</h1>

<pre>
import torchfusion.gan as tfgan
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.cuda as cuda

#Transformations and data augmentation
train_transformations = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 64

# Load the training set
train_set = MNIST(root="./data", train=True, transform=train_transformations, download=True)

train_data = DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=4)

#Create an instance of the NormalDistribution
source = tfgan.NormalDistribution(length=len(train_set),size=(100))
source_data = DataLoader(source,batch_size=batch_size,shuffle=True,num_workers=4)

#Create an instance of the Generator and Discriminator
G = tfgan.MLPGenerator(latent_size=100,output_size=(1,28,28))
D = tfgan.MLPDiscriminator(input_size=(1,28,28))

#Move the networks to GPU if available
if cuda.is_available():
    G.cuda()
    D.cuda()

#Setup the optimizers
g_optim = Adam(G.parameters(),lr=0.0002,betas=(0.5,0.999))
d_optim = Adam(D.parameters(),lr=0.0002,betas=(0.5,0.999))

#Define the loss function
loss_fn = nn.BCELoss()

if __name__ == "__main__":
    #Create an instance of the StandardGANModel
    trainer = tfgan.StandardGANModel(G,D,gen_loss_fn=loss_fn,disc_loss_fn=loss_fn)
    #Train the two models
    trainer.train(train_data,source_data,g_optim,d_optim,num_epochs=200,disc_steps=1,save_interval=3000)
</pre>

<h1>ImageNet Inference</h1>


<pre>

import torchfusion as tf
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models.squeezenet import squeezenet1_1
from PIL import Image

INFER_FOLDER  = r"./images"
MODEL_PATH = r"squeezenet.pth"

transformations = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

infer_set = tf.ImagesFromPaths([INFER_FOLDER],recursive=False,transformations=transformations)
infer_loader = DataLoader(infer_set,batch_size=10)

net = squeezenet1_1()

model = tf.StandardModel(net)
model.load_model(MODEL_PATH)

def predict_loader(data_loader):
    predictions = model.predict(data_loader,apply_softmax=True)
    print(len(predictions))
    for pred in predictions:
        class_index = torch.argmax(pred)
        class_name = tf.decode_imagenet(class_index)
        confidence = torch.max(pred)
        print("Prediction: {} , Accuracy: {} ".format(class_name, confidence))

def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transformations(img)
    img = img.unsqueeze(0)
    pred = model.predict(img,apply_softmax=True)
    class_index = torch.argmax(pred)
    class_name = tf.decode_imagenet(class_index)
    confidence = torch.max(pred)
    print("Prediction: {} , Accuracy: {} ".format(class_name, confidence))


if __name__ == "__main__":
    predict_loader(infer_loader)
    predict_image(r"sample.jpg")



</pre>

<h1>Tutorials</h1>

[Training](TRAINING.md) <br>
[GAN Tutorial](torchfusion/gan)

See more at [Examples](examples)



<h3><b><u>Contact Developers</u></b></h3>
 <p>
  <br>
      <b>John Olafenwa</b> <br>
    <i>Email: </i>    <a style="text-decoration: none;"  href="mailto:johnolafenwa@gmail.com"> johnolafenwa@gmail.com</a> <br>
      <i>Website: </i>    <a style="text-decoration: none;" target="_blank" href="https://john.specpal.science"> https://john.specpal.science</a> <br>
      <i>Twitter: </i>    <a style="text-decoration: none;" target="_blank" href="https://twitter.com/johnolafenwa"> @johnolafenwa</a> <br>
      <i>Medium : </i>    <a style="text-decoration: none;" target="_blank" href="https://medium.com/@johnolafenwa"> @johnolafenwa</a> <br>
      <i>Facebook : </i>    <a style="text-decoration: none;" href="https://facebook.com/olafenwajohn"> olafenwajohn</a> <br>

<br>
  <b>Moses Olafenwa</b> <br>
    <i>Email: </i>    <a style="text-decoration: none;"  href="mailto:guymodscientist@gmail.com"> guymodscientist@gmail.com</a> <br>
      <i>Website: </i>    <a style="text-decoration: none;" target="_blank" href="https://moses.specpal.science"> https://moses.specpal.science</a> <br>
      <i>Twitter: </i>    <a style="text-decoration: none;" target="_blank" href="https://twitter.com/OlafenwaMoses"> @OlafenwaMoses</a> <br>
      <i>Medium : </i>    <a style="text-decoration: none;" target="_blank" href="https://medium.com/@guymodscientist"> @guymodscientist</a> <br>
      <i>Facebook : </i>    <a style="text-decoration: none;" target="_blank" href="https://facebook.com/moses.olafenwa"> moses.olafenwa</a> <br>
<br>
 </p>

 <br>

<pre>

Summary of Resnet50 generated by TorchFusion

Model Summary
Name                      Input Size                Output Size               Parameters                Multiply Adds (Flops)     
Conv2d_1                  [1, 3, 224, 224]          [1, 64, 112, 112]         9408                      118013952                 
BatchNorm2d_1             [1, 64, 112, 112]         [1, 64, 112, 112]         128                       Not Available             
ReLU_1                    [1, 64, 112, 112]         [1, 64, 112, 112]         0                         Not Available             
MaxPool2d_1               [1, 64, 112, 112]         [1, 64, 56, 56]           0                         Not Available             
Conv2d_2                  [1, 64, 56, 56]           [1, 64, 56, 56]           4096                      12845056                  
BatchNorm2d_2             [1, 64, 56, 56]           [1, 64, 56, 56]           128                       Not Available             
ReLU_2                    [1, 64, 56, 56]           [1, 64, 56, 56]           0                         Not Available             
Conv2d_3                  [1, 64, 56, 56]           [1, 64, 56, 56]           36864                     115605504                 
BatchNorm2d_3             [1, 64, 56, 56]           [1, 64, 56, 56]           128                       Not Available             
ReLU_3                    [1, 64, 56, 56]           [1, 64, 56, 56]           0                         Not Available             
Conv2d_4                  [1, 64, 56, 56]           [1, 256, 56, 56]          16384                     51380224                  
BatchNorm2d_4             [1, 256, 56, 56]          [1, 256, 56, 56]          512                       Not Available             
Conv2d_5                  [1, 64, 56, 56]           [1, 256, 56, 56]          16384                     51380224                  
BatchNorm2d_5             [1, 256, 56, 56]          [1, 256, 56, 56]          512                       Not Available             
ReLU_4                    [1, 256, 56, 56]          [1, 256, 56, 56]          0                         Not Available             
Bottleneck_1              [1, 64, 56, 56]           [1, 256, 56, 56]          0                         Not Available             
Conv2d_6                  [1, 256, 56, 56]          [1, 64, 56, 56]           16384                     51380224                  
BatchNorm2d_6             [1, 64, 56, 56]           [1, 64, 56, 56]           128                       Not Available             
ReLU_5                    [1, 64, 56, 56]           [1, 64, 56, 56]           0                         Not Available             
Conv2d_7                  [1, 64, 56, 56]           [1, 64, 56, 56]           36864                     115605504                 
BatchNorm2d_7             [1, 64, 56, 56]           [1, 64, 56, 56]           128                       Not Available             
ReLU_6                    [1, 64, 56, 56]           [1, 64, 56, 56]           0                         Not Available             
Conv2d_8                  [1, 64, 56, 56]           [1, 256, 56, 56]          16384                     51380224                  
BatchNorm2d_8             [1, 256, 56, 56]          [1, 256, 56, 56]          512                       Not Available             
ReLU_7                    [1, 256, 56, 56]          [1, 256, 56, 56]          0                         Not Available             
Bottleneck_2              [1, 256, 56, 56]          [1, 256, 56, 56]          0                         Not Available             
Conv2d_9                  [1, 256, 56, 56]          [1, 64, 56, 56]           16384                     51380224                  
BatchNorm2d_9             [1, 64, 56, 56]           [1, 64, 56, 56]           128                       Not Available             
ReLU_8                    [1, 64, 56, 56]           [1, 64, 56, 56]           0                         Not Available             
Conv2d_10                 [1, 64, 56, 56]           [1, 64, 56, 56]           36864                     115605504                 
BatchNorm2d_10            [1, 64, 56, 56]           [1, 64, 56, 56]           128                       Not Available             
ReLU_9                    [1, 64, 56, 56]           [1, 64, 56, 56]           0                         Not Available             
Conv2d_11                 [1, 64, 56, 56]           [1, 256, 56, 56]          16384                     51380224                  
BatchNorm2d_11            [1, 256, 56, 56]          [1, 256, 56, 56]          512                       Not Available             
ReLU_10                   [1, 256, 56, 56]          [1, 256, 56, 56]          0                         Not Available             
Bottleneck_3              [1, 256, 56, 56]          [1, 256, 56, 56]          0                         Not Available             
Conv2d_12                 [1, 256, 56, 56]          [1, 128, 56, 56]          32768                     102760448                 
BatchNorm2d_12            [1, 128, 56, 56]          [1, 128, 56, 56]          256                       Not Available             
ReLU_11                   [1, 128, 56, 56]          [1, 128, 56, 56]          0                         Not Available             
Conv2d_13                 [1, 128, 56, 56]          [1, 128, 28, 28]          147456                    115605504                 
BatchNorm2d_13            [1, 128, 28, 28]          [1, 128, 28, 28]          256                       Not Available             
ReLU_12                   [1, 128, 28, 28]          [1, 128, 28, 28]          0                         Not Available             
Conv2d_14                 [1, 128, 28, 28]          [1, 512, 28, 28]          65536                     51380224                  
BatchNorm2d_14            [1, 512, 28, 28]          [1, 512, 28, 28]          1024                      Not Available             
Conv2d_15                 [1, 256, 56, 56]          [1, 512, 28, 28]          131072                    102760448                 
BatchNorm2d_15            [1, 512, 28, 28]          [1, 512, 28, 28]          1024                      Not Available             
ReLU_13                   [1, 512, 28, 28]          [1, 512, 28, 28]          0                         Not Available             
Bottleneck_4              [1, 256, 56, 56]          [1, 512, 28, 28]          0                         Not Available             
Conv2d_16                 [1, 512, 28, 28]          [1, 128, 28, 28]          65536                     51380224                  
BatchNorm2d_16            [1, 128, 28, 28]          [1, 128, 28, 28]          256                       Not Available             
ReLU_14                   [1, 128, 28, 28]          [1, 128, 28, 28]          0                         Not Available             
Conv2d_17                 [1, 128, 28, 28]          [1, 128, 28, 28]          147456                    115605504                 
BatchNorm2d_17            [1, 128, 28, 28]          [1, 128, 28, 28]          256                       Not Available             
ReLU_15                   [1, 128, 28, 28]          [1, 128, 28, 28]          0                         Not Available             
Conv2d_18                 [1, 128, 28, 28]          [1, 512, 28, 28]          65536                     51380224                  
BatchNorm2d_18            [1, 512, 28, 28]          [1, 512, 28, 28]          1024                      Not Available             
ReLU_16                   [1, 512, 28, 28]          [1, 512, 28, 28]          0                         Not Available             
Bottleneck_5              [1, 512, 28, 28]          [1, 512, 28, 28]          0                         Not Available             
Conv2d_19                 [1, 512, 28, 28]          [1, 128, 28, 28]          65536                     51380224                  
BatchNorm2d_19            [1, 128, 28, 28]          [1, 128, 28, 28]          256                       Not Available             
ReLU_17                   [1, 128, 28, 28]          [1, 128, 28, 28]          0                         Not Available             
Conv2d_20                 [1, 128, 28, 28]          [1, 128, 28, 28]          147456                    115605504                 
BatchNorm2d_20            [1, 128, 28, 28]          [1, 128, 28, 28]          256                       Not Available             
ReLU_18                   [1, 128, 28, 28]          [1, 128, 28, 28]          0                         Not Available             
Conv2d_21                 [1, 128, 28, 28]          [1, 512, 28, 28]          65536                     51380224                  
BatchNorm2d_21            [1, 512, 28, 28]          [1, 512, 28, 28]          1024                      Not Available             
ReLU_19                   [1, 512, 28, 28]          [1, 512, 28, 28]          0                         Not Available             
Bottleneck_6              [1, 512, 28, 28]          [1, 512, 28, 28]          0                         Not Available             
Conv2d_22                 [1, 512, 28, 28]          [1, 128, 28, 28]          65536                     51380224                  
BatchNorm2d_22            [1, 128, 28, 28]          [1, 128, 28, 28]          256                       Not Available             
ReLU_20                   [1, 128, 28, 28]          [1, 128, 28, 28]          0                         Not Available             
Conv2d_23                 [1, 128, 28, 28]          [1, 128, 28, 28]          147456                    115605504                 
BatchNorm2d_23            [1, 128, 28, 28]          [1, 128, 28, 28]          256                       Not Available             
ReLU_21                   [1, 128, 28, 28]          [1, 128, 28, 28]          0                         Not Available             
Conv2d_24                 [1, 128, 28, 28]          [1, 512, 28, 28]          65536                     51380224                  
BatchNorm2d_24            [1, 512, 28, 28]          [1, 512, 28, 28]          1024                      Not Available             
ReLU_22                   [1, 512, 28, 28]          [1, 512, 28, 28]          0                         Not Available             
Bottleneck_7              [1, 512, 28, 28]          [1, 512, 28, 28]          0                         Not Available             
Conv2d_25                 [1, 512, 28, 28]          [1, 256, 28, 28]          131072                    102760448                 
BatchNorm2d_25            [1, 256, 28, 28]          [1, 256, 28, 28]          512                       Not Available             
ReLU_23                   [1, 256, 28, 28]          [1, 256, 28, 28]          0                         Not Available             
Conv2d_26                 [1, 256, 28, 28]          [1, 256, 14, 14]          589824                    115605504                 
BatchNorm2d_26            [1, 256, 14, 14]          [1, 256, 14, 14]          512                       Not Available             
ReLU_24                   [1, 256, 14, 14]          [1, 256, 14, 14]          0                         Not Available             
Conv2d_27                 [1, 256, 14, 14]          [1, 1024, 14, 14]         262144                    51380224                  
BatchNorm2d_27            [1, 1024, 14, 14]         [1, 1024, 14, 14]         2048                      Not Available             
Conv2d_28                 [1, 512, 28, 28]          [1, 1024, 14, 14]         524288                    102760448                 
BatchNorm2d_28            [1, 1024, 14, 14]         [1, 1024, 14, 14]         2048                      Not Available             
ReLU_25                   [1, 1024, 14, 14]         [1, 1024, 14, 14]         0                         Not Available             
Bottleneck_8              [1, 512, 28, 28]          [1, 1024, 14, 14]         0                         Not Available             
Conv2d_29                 [1, 1024, 14, 14]         [1, 256, 14, 14]          262144                    51380224                  
BatchNorm2d_29            [1, 256, 14, 14]          [1, 256, 14, 14]          512                       Not Available             
ReLU_26                   [1, 256, 14, 14]          [1, 256, 14, 14]          0                         Not Available             
Conv2d_30                 [1, 256, 14, 14]          [1, 256, 14, 14]          589824                    115605504                 
BatchNorm2d_30            [1, 256, 14, 14]          [1, 256, 14, 14]          512                       Not Available             
ReLU_27                   [1, 256, 14, 14]          [1, 256, 14, 14]          0                         Not Available             
Conv2d_31                 [1, 256, 14, 14]          [1, 1024, 14, 14]         262144                    51380224                  
BatchNorm2d_31            [1, 1024, 14, 14]         [1, 1024, 14, 14]         2048                      Not Available             
ReLU_28                   [1, 1024, 14, 14]         [1, 1024, 14, 14]         0                         Not Available             
Bottleneck_9              [1, 1024, 14, 14]         [1, 1024, 14, 14]         0                         Not Available             
Conv2d_32                 [1, 1024, 14, 14]         [1, 256, 14, 14]          262144                    51380224                  
BatchNorm2d_32            [1, 256, 14, 14]          [1, 256, 14, 14]          512                       Not Available             
ReLU_29                   [1, 256, 14, 14]          [1, 256, 14, 14]          0                         Not Available             
Conv2d_33                 [1, 256, 14, 14]          [1, 256, 14, 14]          589824                    115605504                 
BatchNorm2d_33            [1, 256, 14, 14]          [1, 256, 14, 14]          512                       Not Available             
ReLU_30                   [1, 256, 14, 14]          [1, 256, 14, 14]          0                         Not Available             
Conv2d_34                 [1, 256, 14, 14]          [1, 1024, 14, 14]         262144                    51380224                  
BatchNorm2d_34            [1, 1024, 14, 14]         [1, 1024, 14, 14]         2048                      Not Available             
ReLU_31                   [1, 1024, 14, 14]         [1, 1024, 14, 14]         0                         Not Available             
Bottleneck_10             [1, 1024, 14, 14]         [1, 1024, 14, 14]         0                         Not Available             
Conv2d_35                 [1, 1024, 14, 14]         [1, 256, 14, 14]          262144                    51380224                  
BatchNorm2d_35            [1, 256, 14, 14]          [1, 256, 14, 14]          512                       Not Available             
ReLU_32                   [1, 256, 14, 14]          [1, 256, 14, 14]          0                         Not Available             
Conv2d_36                 [1, 256, 14, 14]          [1, 256, 14, 14]          589824                    115605504                 
BatchNorm2d_36            [1, 256, 14, 14]          [1, 256, 14, 14]          512                       Not Available             
ReLU_33                   [1, 256, 14, 14]          [1, 256, 14, 14]          0                         Not Available             
Conv2d_37                 [1, 256, 14, 14]          [1, 1024, 14, 14]         262144                    51380224                  
BatchNorm2d_37            [1, 1024, 14, 14]         [1, 1024, 14, 14]         2048                      Not Available             
ReLU_34                   [1, 1024, 14, 14]         [1, 1024, 14, 14]         0                         Not Available             
Bottleneck_11             [1, 1024, 14, 14]         [1, 1024, 14, 14]         0                         Not Available             
Conv2d_38                 [1, 1024, 14, 14]         [1, 256, 14, 14]          262144                    51380224                  
BatchNorm2d_38            [1, 256, 14, 14]          [1, 256, 14, 14]          512                       Not Available             
ReLU_35                   [1, 256, 14, 14]          [1, 256, 14, 14]          0                         Not Available             
Conv2d_39                 [1, 256, 14, 14]          [1, 256, 14, 14]          589824                    115605504                 
BatchNorm2d_39            [1, 256, 14, 14]          [1, 256, 14, 14]          512                       Not Available             
ReLU_36                   [1, 256, 14, 14]          [1, 256, 14, 14]          0                         Not Available             
Conv2d_40                 [1, 256, 14, 14]          [1, 1024, 14, 14]         262144                    51380224                  
BatchNorm2d_40            [1, 1024, 14, 14]         [1, 1024, 14, 14]         2048                      Not Available             
ReLU_37                   [1, 1024, 14, 14]         [1, 1024, 14, 14]         0                         Not Available             
Bottleneck_12             [1, 1024, 14, 14]         [1, 1024, 14, 14]         0                         Not Available             
Conv2d_41                 [1, 1024, 14, 14]         [1, 256, 14, 14]          262144                    51380224                  
BatchNorm2d_41            [1, 256, 14, 14]          [1, 256, 14, 14]          512                       Not Available             
ReLU_38                   [1, 256, 14, 14]          [1, 256, 14, 14]          0                         Not Available             
Conv2d_42                 [1, 256, 14, 14]          [1, 256, 14, 14]          589824                    115605504                 
BatchNorm2d_42            [1, 256, 14, 14]          [1, 256, 14, 14]          512                       Not Available             
ReLU_39                   [1, 256, 14, 14]          [1, 256, 14, 14]          0                         Not Available             
Conv2d_43                 [1, 256, 14, 14]          [1, 1024, 14, 14]         262144                    51380224                  
BatchNorm2d_43            [1, 1024, 14, 14]         [1, 1024, 14, 14]         2048                      Not Available             
ReLU_40                   [1, 1024, 14, 14]         [1, 1024, 14, 14]         0                         Not Available             
Bottleneck_13             [1, 1024, 14, 14]         [1, 1024, 14, 14]         0                         Not Available             
Conv2d_44                 [1, 1024, 14, 14]         [1, 512, 14, 14]          524288                    102760448                 
BatchNorm2d_44            [1, 512, 14, 14]          [1, 512, 14, 14]          1024                      Not Available             
ReLU_41                   [1, 512, 14, 14]          [1, 512, 14, 14]          0                         Not Available             
Conv2d_45                 [1, 512, 14, 14]          [1, 512, 7, 7]            2359296                   115605504                 
BatchNorm2d_45            [1, 512, 7, 7]            [1, 512, 7, 7]            1024                      Not Available             
ReLU_42                   [1, 512, 7, 7]            [1, 512, 7, 7]            0                         Not Available             
Conv2d_46                 [1, 512, 7, 7]            [1, 2048, 7, 7]           1048576                   51380224                  
BatchNorm2d_46            [1, 2048, 7, 7]           [1, 2048, 7, 7]           4096                      Not Available             
Conv2d_47                 [1, 1024, 14, 14]         [1, 2048, 7, 7]           2097152                   102760448                 
BatchNorm2d_47            [1, 2048, 7, 7]           [1, 2048, 7, 7]           4096                      Not Available             
ReLU_43                   [1, 2048, 7, 7]           [1, 2048, 7, 7]           0                         Not Available             
Bottleneck_14             [1, 1024, 14, 14]         [1, 2048, 7, 7]           0                         Not Available             
Conv2d_48                 [1, 2048, 7, 7]           [1, 512, 7, 7]            1048576                   51380224                  
BatchNorm2d_48            [1, 512, 7, 7]            [1, 512, 7, 7]            1024                      Not Available             
ReLU_44                   [1, 512, 7, 7]            [1, 512, 7, 7]            0                         Not Available             
Conv2d_49                 [1, 512, 7, 7]            [1, 512, 7, 7]            2359296                   115605504                 
BatchNorm2d_49            [1, 512, 7, 7]            [1, 512, 7, 7]            1024                      Not Available             
ReLU_45                   [1, 512, 7, 7]            [1, 512, 7, 7]            0                         Not Available             
Conv2d_50                 [1, 512, 7, 7]            [1, 2048, 7, 7]           1048576                   51380224                  
BatchNorm2d_50            [1, 2048, 7, 7]           [1, 2048, 7, 7]           4096                      Not Available             
ReLU_46                   [1, 2048, 7, 7]           [1, 2048, 7, 7]           0                         Not Available             
Bottleneck_15             [1, 2048, 7, 7]           [1, 2048, 7, 7]           0                         Not Available             
Conv2d_51                 [1, 2048, 7, 7]           [1, 512, 7, 7]            1048576                   51380224                  
BatchNorm2d_51            [1, 512, 7, 7]            [1, 512, 7, 7]            1024                      Not Available             
ReLU_47                   [1, 512, 7, 7]            [1, 512, 7, 7]            0                         Not Available             
Conv2d_52                 [1, 512, 7, 7]            [1, 512, 7, 7]            2359296                   115605504                 
BatchNorm2d_52            [1, 512, 7, 7]            [1, 512, 7, 7]            1024                      Not Available             
ReLU_48                   [1, 512, 7, 7]            [1, 512, 7, 7]            0                         Not Available             
Conv2d_53                 [1, 512, 7, 7]            [1, 2048, 7, 7]           1048576                   51380224                  
BatchNorm2d_53            [1, 2048, 7, 7]           [1, 2048, 7, 7]           4096                      Not Available             
ReLU_49                   [1, 2048, 7, 7]           [1, 2048, 7, 7]           0                         Not Available             
Bottleneck_16             [1, 2048, 7, 7]           [1, 2048, 7, 7]           0                         Not Available             
AvgPool2d_1               [1, 2048, 7, 7]           [1, 2048, 1, 1]           0                         Not Available             
Linear_1                  [1, 2048]                 [1, 1000]                 2049000                   2048000                   

Total Parameters: 25557032
Total Multiply Adds (For Convolution aand Linear Layers only): 4089184256
Number of Layers
MaxPool2d : 1 layers   Linear : 1 layers   AvgPool2d : 1 layers   Bottleneck : 16 layers   ReLU : 49 layers   Conv2d : 53 layers   BatchNorm2d : 53 layers 


</pre>
















