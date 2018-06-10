import torchfusion as tf
from torchvision.datasets.cifar import CIFAR10
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torch.nn as nn
from torch.optim import Adam
import torch.cuda as cuda


#Unit Module to be used in the main network
class Unit(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Unit,self).__init__()

        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self,inputs):
        outputs = self.conv(inputs)
        outputs = self.bn(outputs)
        outputs = self.relu(outputs)

        return outputs

#Define a the classifier network
class CifarNet(nn.Module):
    def __init__(self):
        super(CifarNet,self).__init__()

        self.net = nn.Sequential(
            Unit(3,64),
            Unit(64,64),
            Unit(64, 64),

            nn.MaxPool2d(kernel_size=3,stride=2),

            Unit(64, 128),
            Unit(128, 128),
            Unit(128, 128),

            nn.MaxPool2d(kernel_size=3, stride=2),

            Unit(128, 256),
            Unit(256, 256),
            Unit(256, 256),

            tf.GlobalAvgPool2d(),

            nn.Linear(256,10)

        )

    def forward(self,inputs):

        outputs = self.net(inputs)

        return outputs


batch_size = 64

#Transformations and data augmentation
transformations = transforms.Compose([
    transforms.RandomCrop(32,padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

#Load the training and test sets
train_set = CIFAR10(root="./data",transform=transformations,download=True)
test_set = CIFAR10(root="./data",train=False,transform=transformations,download=True)

train_loader = DataLoader(train_set,shuffle=True,batch_size=batch_size,num_workers=4)
test_loader = DataLoader(test_set,shuffle=False,batch_size=batch_size,num_workers=4)

#Create an instance of the network
net = CifarNet()

#Move to GPU if available
if cuda.is_available():
    net.cuda()

#Setup the optimize and a loss function
optimizer = Adam(net.parameters(),lr=0.001)
loss_fn = nn.CrossEntropyLoss()

#Top 1 Train accuracy
train_metrics = tf.Accuracy(topK=1)

#Top 1 and Top 2 train accuracy
test_metrics_top1 = tf.Accuracy(name="Top 1 Acc ",topK=1)
test_metrics_top2 = tf.Accuracy(name="Top 2 Acc ",topK=2)

model = tf.StandardModel(net)

#Define a learning rate schedule function
def lr_schedule(e):

    lr = 0.001

    if e > 90:
        lr /= 1000
    elif e > 60:
        lr /= 100
    elif e > 30:
        lr /= 10
    print("LR: ",lr)
    return lr

def train():

    #Print a summary of the network
    print(model.summary((3,32,32)))

    """
        Train the network for 120 epochs
        If in jupyter notebook, set notebook_mode=True for live progress
        you can disable batch_log if you don't need batch_level logging

        """
    model.train(train_loader, loss_fn, optimizer, [train_metrics], test_loader,
                [test_metrics_top1, test_metrics_top2], num_epochs=120,
                model_dir="cifar10_saved_models",notebook_mode=False,batch_log=False,save_logs="logs.txt",lr_schedule=lr_schedule)


if __name__ == "__main__":
    
    # always initiate training in the main loop
    train()
