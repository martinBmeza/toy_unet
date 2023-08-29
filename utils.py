import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio
from torchvision.utils import make_grid

def compare_spectrogram(output, _input, target, plot=False, title=None):
    img_grid = make_grid(
            torch.cat((output, _input, target), dim=0), 
            scale_each=True, 
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
