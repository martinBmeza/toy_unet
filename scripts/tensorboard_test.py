"""
Thougth to be runned from the root directory.
Moved here for organization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from dataset import DummyDataset
from utils import compare_spectrogram
from torch.utils.tensorboard import SummaryWriter
from torchaudio.transforms import Spectrogram

spectrogram = Spectrogram(n_fft=512, center=False, normalized=True)
dataset = DummyDataset('data/', transform=spectrogram)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
dataiter = iter(dataloader)
noisy, clean = next(dataiter)
img = compare_spectrogram(clean, noisy) 

# init writer with log dir
writer = SummaryWriter('runs/unet_denoising_experiment')

#write data to Tensorboard log dir
writer.add_image('Example image', img, dataformats='CHW')
writer.flush()




