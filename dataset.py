import torch
import torchaudio
import os
import soundfile as sf
from torch.utils.data import DataLoader, Dataset
from torchaudio.transforms import Spectrogram
from glob import glob
from tqdm import tqdm



class DummyDataset(Dataset):
    def __init__(self, root_dir, transform='spectrogram'):
        self.root_dir = root_dir
        if transform == 'spectrogram':
            self.transform = Spectrogram(n_fft=512, center=False, normalized=True)
        else:
            raise TypeError('not supported transform!')
            exit()
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

class DummyDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(DummyDataLoader, self).__init__(*args, **kwargs)

if __name__ == '__main__':
    from utils import plot_spectrogram
    dataset = DummyDataset('data/train/')
    dataloader = DummyDataLoader(dataset, batch_size=8)

    for batch in dataloader:
        x, y = batch
        print(x.shape, y.shape)
        plot_spectrogram(x[0])
        break
