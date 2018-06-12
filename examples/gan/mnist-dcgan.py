import torchfusion.gan as tfgan
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.cuda as cuda


train_transformations = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 64
# Load the training set
train_set = MNIST(root="./data", train=True, transform=train_transformations, download=True)

train_data = DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=4)


source = tfgan.NormalDistribution(length=len(train_set),size=(100,1,1))
source_data = DataLoader(source,batch_size=batch_size,shuffle=True,num_workers=4)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

G = tfgan.DCGANGenerator(latent_size=(100,1,1),output_size=(1,32,32))
D = tfgan.DCGANDiscriminator(input_size=(1,32,32))


if cuda.is_available():
    G.cuda()
    D.cuda()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

G.apply(weights_init)
D.apply(weights_init)

g_optim = Adam(G.parameters(),lr=0.0002,betas=(0.5,0.999))
d_optim = Adam(D.parameters(),lr=0.0002,betas=(0.5,0.999))

loss_fn = nn.BCELoss()

if __name__ == "__main__":
    trainer = tfgan.StandardGANModel(G,D,gen_loss_fn=loss_fn,disc_loss_fn=loss_fn)
    trainer.train(train_data,source_data,g_optim,d_optim,num_epochs=200,disc_steps=1,save_interval=3000,notebook_mode=False,batch_log=True)
