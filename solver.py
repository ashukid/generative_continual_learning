
# coding: utf-8

# In[4]:


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

from utils import plot_images


# In[5]:


class dataset(torch.utils.data.Dataset):
    
    def __init__(self,xdata,ydata):
        self.xdata=xdata
        self.ydata=ydata
        
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
        image=self.to_sigmoid(self.xdata[idx,:]).astype(np.float32)
        label=self.ydata[idx].astype(np.int64)
        return image,label
    
    def __len__(self):
        return len(self.xdata)


# In[7]:


def load_dataset(xdata,ydata,batch_size):
    dset=dataset(xdata,ydata)
    dloader=torch.utils.data.DataLoader(dset,batch_size=batch_size,shuffle=True)
    
    return dloader


# In[8]:


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv=nn.Sequential(
            
            #28,28,1
            nn.Conv2d(1,8,3,stride=1,padding=0),
            nn.Conv2d(8,8,3,stride=2,padding=0),
            nn.ReLU(inplace=True),
            
            #12,12,8
            nn.Conv2d(8,16,3,stride=1,padding=0),
            nn.Conv2d(16,16,3,stride=2,padding=0),
            nn.ReLU(inplace=True),#4,4,16
        )
        
        self.fc=nn.Sequential(
            
            #256
            nn.Linear(256,500),
            nn.ReLU(inplace=True),
            nn.Linear(500,10),
        )
        
    def forward(self,inp):
        feat=self.conv(inp).view(-1,256)
        out=self.fc(feat)
        return out


# In[9]:


cuda=torch.cuda.is_available()
device=torch.device("cuda:0" if cuda else "cpu")

net=Network().to(device)
criterion=nn.CrossEntropyLoss()
optim=torch.optim.Adam(net.parameters(),lr=0.001)


# In[17]:


def train(dloader,epoch,path):
    
    net.train()
    for epoch in range(epoch):
        iter_loss=0
        for x,y in dloader:
            x,y=x.to(device),y.to(device)
            output=net(x)
            loss=criterion(output,y)
            iter_loss+=loss.item()

            net.zero_grad()
            loss.backward()
            optim.step()
        print(f"Epoch : {epoch} Loss : {iter_loss}")
        
    #code to save the network
    torch.save(net.state_dict(),path)


# In[18]:


def main(xdata,ydata,batch_size,epoch,path):
    dloader=load_dataset(xdata,ydata,batch_size)
    train(dloader,epoch,path)

