"""
TODO : 
    GPU parametrized control
    visualize data flow within the model 
    Save model weigths every n epochs / training steps
    training scheduller
"""
import torch
import numpy as np
from utils import compare_spectrogram
from model import UNET
from dataset import DummyDataset
from torchaudio.transforms import Spectrogram
from torch.utils.tensorboard import SummaryWriter

# tensorboard
writer = SummaryWriter('runs/UNET_experiment_badly_named')

# model
model = UNET()

# dataset
spectrogram = Spectrogram(n_fft=512, center=False, normalized=True)
train_dataset = DummyDataset('data/train/', transform=spectrogram)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
val_dataset = DummyDataset('data/val/', transform=spectrogram)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

# loss and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# training loop
epochs = 10
logger_batch_freq = 8
for epoch in range(1, epochs+1):
    running_loss = 0.0
    print(f"epoch: {epoch}/{epochs}")
    for batch_idx, (x, y) in enumerate(train_dataloader):
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % logger_batch_freq == (logger_batch_freq-1):
            running_vloss = 0.0 # check in validation set 
            model.train(False)  # no need to track gradients
            for i, (xval, yval) in enumerate(val_dataloader):
                outputs = model(xval)
                vloss = criterion(outputs, yval)
                running_vloss += vloss.item()
            avg_loss  = running_loss / logger_batch_freq
            avg_vloss = running_vloss / len(val_dataloader) 
            # log the running losses
            writer.add_scalars("Training vs Validation Loss",
                    {'Training':avg_loss, 'Validation':avg_vloss},
                    epoch * len(train_dataloader) + batch_idx)
            running_loss = 0.0
            # log a sample
            sample=int(np.random.choice(range(len(val_dataloader)), 1))
            img = compare_spectrogram(
                    outputs[sample:sample+1].detach(), 
                    xval[sample:sample+1].detach(),
                    yval[sample:sample+1].detach())
            writer.add_image(f"input_output_target_e{epoch}_b{batch_idx}", img, dataformats='CHW')
            model.train(True) # turn gradients back on for training
        loss, current = loss.item(), (batch_idx+1)*len(x)
        print(f"loss: {loss:.4f} [{current:>5d}/{len(train_dataset):>5d}]")

print("Finished Training")
writer.flush()

        
