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

def plot_images(x):
    plt.figure(figsize=(5,5))
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(x[i].squeeze())
        
