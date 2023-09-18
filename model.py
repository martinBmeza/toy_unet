import torch
import os
import soundfile as sf
import torch.nn as nn
from torch.utils.data import Dataset
from glob import glob
from tqdm import tqdm


class MLP(nn.Module):
    def __init__(self, dim_in=16000, hidden_units=100, dim_out=16000):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(dim_in, hidden_units)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_units, dim_out)

    def forward(self, inputs):
        x = self.fc1(inputs)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class encoder_block(nn.Module):
    def __init__(self, channels_in, channels_out, final=False):
        super().__init__()
        self.final = final
        self.conv = nn.Conv2d(channels_in, channels_out, kernel_size=4, stride=2, padding=1)
        if final:
            self.tanh = nn.Tanh()
        else:
            self.norm = nn.BatchNorm2d(channels_out)
            self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv(inputs)
        if self.final:
            x = self.tanh(x)
        else:
            x = self.norm(x)
            x = self.relu(x)
        return x 


class decoder_block(nn.Module):
    def __init__(self, channels_in, channels_out, final=False):
        super().__init__()
        self.final = final
        self.conv = nn.ConvTranspose2d(channels_in, channels_out, kernel_size=4, stride=2, padding=1)
        self.relu = nn.ReLU()
        if not final:
            self.norm = nn.BatchNorm2d(channels_out)

    def forward(self, inputs, skip=None):
        x = self.conv(inputs)
        if self.final:
            x = self.relu(x)
            return x
        else:
            x = self.norm(x)
            x = self.relu(x)
        return torch.cat([x, skip], axis=1) 


class UNET(nn.Module):
    """
    Add dimensions configurable params to simulate experiments
    """
    def __init__(self):
        super().__init__()
        # encoder                         # dim out
        self.enc1 = encoder_block(1,2)    # 128 x 128
        self.enc2 = encoder_block(2,4)    # 64 x 64
        self.enc3 = encoder_block(4,8)    # 32 x 32
        self.enc4 = encoder_block(8,16)   # 16 x 16
        self.enc5 = encoder_block(16,32)  # 8 x 8
        self.enc6 = encoder_block(32,64)  # 4 x 4
        self.enc7 = encoder_block(64,128) # 2 x 2

        # bottleneck
        self.b = encoder_block(128, 256, final=True) # 1 x 1

        # decoder
        self.dec1 = decoder_block(256, 128) # 2 x 2
        self.dec2 = decoder_block(128 + 128, 64)  # 4 x 4
        self.dec3 = decoder_block(64 + 64, 32)   # 8 x 8
        self.dec4 = decoder_block(32 + 32, 16)   # 16 x 16
        self.dec5 = decoder_block(16 + 16, 8)    # 32 x 32
        self.dec6 = decoder_block(8 + 8, 4)     # 64 x 64
        self.dec7 = decoder_block(4 + 4, 2)     # 128 x 128 
        
        # output 
        self.f = decoder_block(2 + 2, 1, final=True) # 256 x 256

    def forward(self, inputs):
        e1 = self.enc1(inputs)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)

        b  = self.b(e7)

        d1 = self.dec1(b, e7)
        d2 = self.dec2(d1, e6)
        d3 = self.dec3(d2, e5)
        d4 = self.dec4(d3, e4)
        d5 = self.dec5(d4, e3)
        d6 = self.dec6(d5, e2)
        d7 = self.dec7(d6, e1)

        output = self.f(d7)
        output_pad = nn.functional.pad(output, (0,0, 0, 1), 'constant', output.min().item()) 
        return output_pad

if __name__ == '__main__':
    inputs = torch.randn((2, 1, 257, 256))
    model = UNET()
    y = model(inputs)
    print(y.shape)
