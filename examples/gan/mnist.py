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
    trainer.train(train_data,source_data,g_optim,d_optim,num_epochs=200,disc_steps=1,save_interval=3000,notebook_mode=False,batch_log=True)