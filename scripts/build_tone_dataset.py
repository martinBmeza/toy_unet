import argparse
import librosa 
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
from tqdm import tqdm


def ratio_type(arg):
    try:
        f = float(arg)
    except ValueError:
        raise argparse.ArgumentTypeError("Must be a floating point number")
    if (f<0.0) or (f>1.0):
        raise argparse.ArgumentTypeError("Argument must be between 0 and 1")
    return f

parser = argparse.ArgumentParser(description='Build a dummy tone database')
parser.add_argument('nsamples', metavar='N', type=int)
parser.add_argument('dur_samples', metavar='Dur', type=int)
parser.add_argument('sr', metavar='SR', type=int)
parser.add_argument('fmin', metavar='Fmin', type=int)
parser.add_argument('fmax', metavar='Fmax', type=int)
parser.add_argument('noise_ratio', metavar='Noise', type=ratio_type)
parser.add_argument('split_ratio', metavar='Split', type=ratio_type)
args = parser.parse_args()
info = (f"\nCreating a dummy database of {args.nsamples}\n"
        f"composed by tones of {args.dur_samples} samples  @ {args.sr} Hz\n"
        f"with sinus from {args.fmin} Hz to {args.fmax} Hz\n"
        f"and a noise ratio of {args.noise_ratio}.\n"
        f"The {args.split_ratio:.2%} is selected for train, and the remainder\n"
        f"is asigned equally for validation and test\n")
print(info)

x = np.linspace(0, args.dur_samples//args.sr, args.dur_samples, endpoint=False)
clean_tones = [np.sin(2*np.pi*f*x) for f in np.random.uniform(args.fmin, args.fmax, args.nsamples)]
noisy_tones = [(clean_tone*(1-args.noise_ratio))+(np.random.normal(0, 1, x.size)*(args.noise_ratio)) for clean_tone in clean_tones]
noisy_tones = [noisy_tone / np.max(np.abs(noisy_tone)) for noisy_tone in noisy_tones]

# -----------------------------------------------------
# |    TRAIN            VAL            TEST           |
# | 0 ------- split_1 ------- split_2 ------- nsamples|
# -----------------------------------------------------

split_1 = int(args.nsamples*args.split_ratio)
split_2 = split_1 + ((args.nsamples - split_1)//2)
train_idxs = range(split_1)
val_idxs   = range(split_1, split_2)
test_idxs  = range(split_2, args.nsamples)
for tone_idx in tqdm(train_idxs):
    sf.write(f"data/train/clean/clean_{tone_idx:04.0f}.wav", clean_tones[tone_idx], args.sr)
    sf.write(f"data/train/noisy/noisy_{tone_idx:04.0f}.wav", noisy_tones[tone_idx], args.sr)
for tone_idx in tqdm(val_idxs):
    sf.write(f"data/val/clean/clean_{tone_idx:04.0f}.wav", clean_tones[tone_idx], args.sr)
    sf.write(f"data/val/noisy/noisy_{tone_idx:04.0f}.wav", noisy_tones[tone_idx], args.sr)
for tone_idx in tqdm(test_idxs):
    sf.write(f"data/test/clean/clean_{tone_idx:04.0f}.wav", clean_tones[tone_idx], args.sr)
    sf.write(f"data/test/noisy/noisy_{tone_idx:04.0f}.wav", noisy_tones[tone_idx], args.sr)

print(f"train = {split_1}\nval  = {split_2 - split_1}\ntest = {args.nsamples-split_2}")
# plot some examples
tone_index = np.random.choice(range(args.nsamples), 1)[0]
stft_params = {
        'n_fft' : 512,
        'hop_length' : 128,
        'win_length' : 512
        }
clean_mag = librosa.amplitude_to_db(abs(librosa.stft(clean_tones[tone_index], **stft_params)), ref=np.max)
noisy_mag = librosa.amplitude_to_db(abs(librosa.stft(noisy_tones[tone_index], **stft_params)), ref=np.max)

display_params = {
        'y_axis' : 'log',
        'sr' : args.sr,
        'hop_length' : stft_params['hop_length'],
        'x_axis' : 'time'
        }
plt.subplot(2,1,1)
plt.title('Clean signal (target)')
librosa.display.specshow(clean_mag, **display_params)
plt.subplot(2,1,2)
plt.title('Noisy signal (input)')
librosa.display.specshow(noisy_mag,**display_params)
plt.subplots_adjust(hspace=0.5)
plt.show()
