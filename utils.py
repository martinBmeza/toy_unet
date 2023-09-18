import torch
import sys, os
import logging
import librosa
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio
from torchvision.utils import make_grid
from safe_gpu import safe_gpu

def setup_gpu(n=1):
    safe_gpu.claim_gpus(n)
    print(f"Allocatd devices: {safe_gpu.gpu_owner.devices_taken}")
    if torch.cuda.is_available():
        print("cuda is available")
    else:
        print("cuda is not available")
        exit()
    return safe_gpu

def create_logger(save_folder, log_level=logging.INFO): 
    os.makedirs(save_folder, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    log_formatter = logging.Formatter("%(asctime)s [ %(levelname)-5.5s]:  %(message)s")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    filepath = os.path.join(save_folder,'out.log')
    if os.path.exists(filepath):
        os.remove(filepath)
    file_handler = logging.FileHandler(filepath)
    file_handler.setFormatter(log_formatter)
    if (root_logger.hasHandlers()):
        root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    return root_logger



def compare_spectrogram(output, _input, target, plot=False, title=None):
    #import pdb; pdb.set_trace()
    img_grid = make_grid(
            torch.cat((output, _input, target), dim=0), 
            scale_each=False, 
            padding=0)
    img_grid = img_grid.mean(dim=0).numpy()
    img_grid = librosa.power_to_db(img_grid)
    if plot:
        fig, axs = plt.subplots(1, 1)
        axs.set_title(title or "Spectrogram (db)")
        axs.set_ylabel("freq_bin")
        axs.set_xlabel("frame")
        im = axs.imshow(img_grid, origin="lower", aspect="auto")
        fig.colorbar(im, ax=axs)
        plt.show()
    return scale_minmax(img_grid)[np.newaxis,:]

def scale_minmax(X, _min=0, _max=255):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled =  X_std * (_max - _min) + _min
    X_int = X_scaled.astype(np.uint8)
    X = 255 - X_int
    return X

def plot_waveform(waveform, sr, title="Waveform"):
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr
    figure, axes = plt.subplots(num_channels, 1)
    axes.plot(time_axis, waveform[0], linewidth=1)
    axes.grid(True)
    figure.suptitle(title)
    plt.show(block=False)


def plot_spectrogram(specgram, title=None, ylabel="freq_bin"):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    specgram = specgram[0][0].numpy()
    im = axs.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)
    plt.show()


def plot_fbank(fbank, title=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Filter bank")
    axs.imshow(fbank, aspect="auto")
    axs.set_ylabel("frequency bin")
    axs.set_xlabel("mel bin")
    plt.show(block=False)

if __name__ == '__main__':
    print('go!')
