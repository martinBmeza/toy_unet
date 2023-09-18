"""
TODO:
    migrate config to a .py file for parameter flexibility
"""
import os
import argparse
import torch
from safe_gpu import safe_gpu
from solver import Solver
from utils import setup_gpu, create_logger, compare_spectrogram
from model import UNET
from dataset import DummyDataset, DummyDataLoader
from torchaudio.transforms import Spectrogram
from torch.utils.tensorboard import SummaryWriter



parser = argparse.ArgumentParser(
        "UNET tone denoiser"
        "0-->No / 1-->Yes")
### General Config
## task related
parser.add_argument('--name', type=str, default='test_exp', help='Name of the experiment')
parser.add_argument('--train_data_dir', type=str, default='data/train', help='directory including clean and noisy pair folders for train')
parser.add_argument('--val_data_dir', type=str, default='data/val', help='directory including clean and noisy pair folder for validation')
## network architeture
#
## training config
parser.add_argument('--use_cuda', type=int, default=1, help='Wheter use GPU (0-->No / 1-->Yes)')
parser.add_argument('--epochs', type=int, default=100, help='Number of maximum epochs')
parser.add_argument('--half_lr', type=int, default=0, help='Halving learning rate when get small improvement')
parser.add_argument('--early_stop', type=int, default=0, help='Early stop training when no improvement for 10 epochs')
parser.add_argument('--max_norm', type=float, default=5, help='Gradient norm treshold to clip')
# minibatch
parser.add_argument('--shuffle', type=int, default=0, help='re-shuffle the data at every epoch')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--num_workers', type=int, default=8, help='Number of workers to generate minibatch')
# optimizer and loss
parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam'], help='Optimizer (only support adam and sgd)')
parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'l1'], help='Loss function (only support mse)')
parser.add_argument('--lr', type=float, default=0.125, help='Init learning rate')
parser.add_argument('--momentum', type=float, default=0.0, help='Momentum for optimizer')
parser.add_argument('--l2', type=float, default=0, help='Weight decay (L2 penalty)')
# save and load model
parser.add_argument('--save_folder', type=str, default='exp/temp', help='Location to save epoch models')
parser.add_argument('--checkpoint', type=int, default=0, help='Enables checkpoint saving of model')
parser.add_argument('--continue_from', type=str, default='', help='Continue from checkpoinr model')
parser.add_argument('--model_path', type=str, default='final.pth.tar', help='Location to save best validated model')
# logger
parser.add_argument('--print_freq', type=int, default=1000, help='Frequency of printing training information')
parser.add_argument('--tboard_epoch', type=int, default=0, help='Turn on tensorboard logger information')
parser.add_argument('--tboard_graph', type=int, default=0, help='Turn on tensorboard graphs')


def main(logger, args):
    # data
    train_dataset = DummyDataset(args.train_data_dir)
    val_dataset = DummyDataset(args.val_data_dir)
    train_dataloader = DummyDataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)
    val_dataloader = DummyDataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    data = {'train_dataloader' : train_dataloader, 'val_dataloader' : val_dataloader}

    # model
    model = UNET()
    k = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'{k} of parameters')
    if args.use_cuda:
        gpus = [str(g) for g in safe_gpu.gpu_owner.devices_taken]
        os.environ['CUDA_VISIBLE_DEVICES']=','.join(gpus)
        model = torch.nn.DataParallel(model)
        model.cuda()

    # optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    else:
        raise ValueError('Not support optimizer')


    # loss
    if args.loss == 'mse':
        criterion = torch.nn.MSELoss()
    else:
        raise ValueError('Not support loss')

    # solver
    solver = Solver(logger, data, model, optimizer, criterion, args)
    solver.train()

if __name__ == '__main__':
    args = parser.parse_args()
    setup_gpu()
    logger = create_logger(os.path.join(args.save_folder, args.name))
    logger.info('Start!')
    logger.info(args)
    main(logger, args)

#
#
#
#
## tensorboard
#writer = SummaryWriter('runs/UNET_experiment_badly_named')
#
## model
#model = UNET()
#model = model.cuda()
## dataset
#spectrogram = Spectrogram(n_fft=512, center=False, normalized=True)
#train_dataset = DummyDataset('data/train/', transform=spectrogram)
#train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256)
#val_dataset = DummyDataset('data/val/', transform=spectrogram)
#val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=256)
#
## loss and optimizer
#optimizer = torch.optim.Adam(model.parameters())
#
## training loop
#epochs = 100
#logger_batch_freq = 20
#for epoch in range(1, epochs+1):
#    # train
#    running_loss = 0.0
#    print(f"epoch: {epoch}/{epochs}")
#    model.train()
#    for batch_idx, (x, y) in enumerate(train_dataloader):
#        x, y = x.cuda(), y.cuda()
#        optimizer.zero_grad()
#        outputs = model(x)
#        loss = criterion(outputs, y)
#        loss.backward()
#        optimizer.step()
#        running_loss += loss.item()
#
#       # # eval
#       # if batch_idx % logger_batch_freq == (logger_batch_freq-1):
#       #     model.eval()
#       #     running_vloss = 0.0 # check in validation set
#       #     for i, (xval, yval) in enumerate(val_dataloader):
#       #         xval, yval = xval.cuda(), yval.cuda()
#       #         outputs = model(xval)
#       #         vloss = criterion(outputs, yval)
#       #         running_vloss += vloss.item()
#       #     avg_loss  = running_loss / logger_batch_freq
#       #     avg_vloss = running_vloss / len(val_dataloader)
#       #     # log the running losses
#       #     writer.add_scalars("Training vs Validation Loss",
#       #             {'Training':avg_loss, 'Validation':avg_vloss},
#       #             epoch * len(train_dataloader) + batch_idx)
#       #     running_loss = 0.0
#       #     # log a sample
#       #     #sample=int(np.random.choice(range(len(val_dataloader)), 1))
#       #     #img = compare_spectrogram(
#       #     #        outputs[sample:sample+1].cpu().detach(),
#       #     #        xval[sample:sample+1].cpu().detach(),
#       #     #        yval[sample:sample+1].cpu().detach())
#       #     #writer.add_image(f"input_output_target_e{epoch}_b{batch_idx}", img, dataformats='CHW')
#        loss, current = loss.item(), (batch_idx+1)*len(x)
#        print(f"loss: {loss:.4f} [{current:>5d}/{len(train_dataset):>5d}]")
#
#print("Finished Training")
#writer.flush()
#
