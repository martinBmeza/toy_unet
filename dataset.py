import torch
import torchaudio
import os
import soundfile as sf
from torch.utils.data import Dataset
from torchaudio.transforms import Spectrogram
from glob import glob
from tqdm import tqdm



class DummyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.clean_paths = glob(os.path.join(root_dir,'clean','**.wav'), recursive=True)
        self.clean_paths.sort()
        self.noisy_paths = glob(os.path.join(root_dir,'noisy','**.wav'), recursive=True)
        self.noisy_paths.sort()

    def __len__(self):
        return len(self.clean_paths)

    def __getitem__(self, idx):
        noisy, _ = torchaudio.load(
                self.noisy_paths[idx], 
                normalize=True)
        clean, _ = torchaudio.load(
                self.clean_paths[idx], 
                normalize=True)
        if self.transform:
            noisy = self.transform(noisy)
            clean = self.transform(clean)
        return noisy, clean

class STFT_power(object):
    def __init__(self, win_len, hop_size, sr):
        assert all(isinstance(i, int) for i in [win_len, hop_size, sr])
        self.win_len = win_len
        self.hop_size = hop_size
        self.sr = sr

    def __call__(self, sample):
        pass

if __name__ == '__main__':
    from utils import plot_spectrogram
    #transforms.Compose
    spectrogram = Spectrogram(
            n_fft=512, 
            center=False, 
            normalized=True)
    dataset = DummyDataset('data/train/', transform=spectrogram)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)

    for batch in dataloader:
        x, y = batch
        print(x.shape, y.shape)
        plot_spectrogram(x[0])
        break
