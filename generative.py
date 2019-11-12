
# coding: utf-8

# wgan gp for mnist dataset

# In[1]:


import numpy as np
import pandas as pd
import os
import shutil
import glob
import matplotlib.pyplot as plt
import random
import torch
from torch import nn
import torchvision
import torch.utils.data
from torchvision.utils import save_image
import torch.optim as optim
from PIL import Image


# In[2]:


class dataset(torch.utils.data.Dataset):
    
    def __init__(self,data):
        self.data=data
        
    def to_tanh(self,image):
        upper=image.max()
        lower=image.min()
        image=2*((image-lower)/(upper-lower))-1
        return image
        
    def to_sigmoid(self,image):
        upper=image.max()
        lower=image.min()
        image=(image-lower)/(upper-lower)
        return image
        
    def __getitem__(self,idx):
        image=self.to_tanh(self.data[idx,:]).astype(np.float32)
        return image
    
    def __len__(self):
        return len(self.data)


# In[3]:


def load_dataset(data,batch_size):
    dset=dataset(data)
    dloader=torch.utils.data.DataLoader(dset,batch_size=batch_size,shuffle=True)
    
    return dloader


# In[4]:


class DNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.main=nn.Sequential(
            
            #28,28,1
            nn.Conv2d(1,8,3,stride=1,padding=0),
            nn.Conv2d(8,8,3,stride=2,padding=0),
            nn.LeakyReLU(0.2,inplace=True),
            
            #12,12,8
            nn.Conv2d(8,16,3,stride=1,padding=0),
            nn.Conv2d(16,16,3,stride=2,padding=0),
            nn.LeakyReLU(0.2,inplace=True),
            
            #4,4,16
            nn.Conv2d(16,1,4,stride=1,padding=0),
            nn.Sigmoid()#1,1,1
        )
        
    def forward(self,inp):
        out=self.main(inp)
        return out.view(-1,1).squeeze()


# In[5]:


class GNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.main=nn.Sequential(
        
            #1,1,50
            nn.ConvTranspose2d(50,16,kernel_size=4,stride=1),
            nn.ReLU(inplace=True),
            
            #4,4,16
            nn.ConvTranspose2d(16,8,kernel_size=3,stride=3),
            nn.ReLU(inplace=True),
            
            #12,12,8
            nn.ConvTranspose2d(8,1,kernel_size=6,stride=2),
            nn.Tanh(),#28,28,1
        )
        
    def forward(self,inp):
        out=self.main(inp)
        return out


# In[6]:


cuda=torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

netG=GNetwork().to(device)
netD=DNetwork().to(device)

doptim=torch.optim.Adam(netD.parameters(),lr=0.001,betas=(0,0.9))
goptim=torch.optim.Adam(netG.parameters(),lr=0.001,betas=(0,0.9))


# In[11]:


def train(dloader,epoch,gpath,dpath):

    netG.train()
    netD.train()
    
    for epoch in range(epoch):
        gen_iter_loss=0
        dis_iter_loss=0

        for x in dloader:

            real=x.to(device)
            z=torch.randn(x.size(0), 50, 1, 1, device=device)
            fake=netG(z)

            #training discriminator
            netD.zero_grad()
            dloss=-torch.mean(torch.log(netD(real))+torch.log(1-netD(fake.detach())))
            dloss.backward()
            dis_iter_loss+=dloss.item()
            doptim.step()

            #training generator
            netG.zero_grad()
            gloss=-torch.mean(torch.log(netD(fake)))
            gloss.backward()
            gen_iter_loss+=gloss.item()
            goptim.step()


        print(f"DISCRIMNATOR LOSS : {dis_iter_loss} GENERATOR LOSS : {gen_iter_loss}")
        
    torch.save(netD.state_dict(),dpath)
    torch.save(netG.state_dict(),gpath)


# In[18]:


def main(data,batch_size,epoch,gpath,dpath):
    dloader=load_dataset(data,batch_size)
    train(dloader,epoch,gpath,dpath)

