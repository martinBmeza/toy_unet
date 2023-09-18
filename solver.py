import os
import time
import torch
import numpy as np

class Solver(object):
    def __init__(self, logger, data, model, optimizer, criterion, args):
        self.logger = logger
        self.train_dataloader = data['train_dataloader']
        self.val_dataloader = data['val_dataloader']
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        # training config
        self.use_cuda = args.use_cuda
        self.epochs = args.epochs
        self.early_stop = args.early_stop
        self.max_norm = args.max_norm
        self.half_lr = args.half_lr

        # save and load model
        self.save_folder = os.path.join(args.save_folder, args.name)
        self.checkpoint = args.checkpoint
        self.continue_from = args.continue_from
        self.model_path = args.model_path

        # self.logger (logger / tensorboard)
        self.print_freq = args.print_freq
        # todo: add tensorboard handles
        #if self.tensorboard:
        #    from torch.utils.tensorboard import SummaryWriter
        #    self.writer = SummaryWriter(os.path.join('runs', args.name))

        # reset
        self._reset()

    def _reset(self):
        if self.continue_from:
            self.logger.info(f'loading checkpoint model {self.continue_from}')
            cont = torch.load(self.continue_from)
            self.start_epoch = cont['epoch']
            self.model.load_state_dict(cont['model_state_dict'])
            self.optimizer.load_state_dict(cont['optimizer_state'])
            torch.set_rng_state(cont['trandom_state'])
            np.random.set_state(cont['nrandom_state'])
        else:
            self.start_epoch = 0
        os.makedirs(self.save_folder, exist_ok=True)
        self.prev_val_loss = float('inf')
        self.best_val_loss = float('inf')
        self.halving = False
        self.val_no_impv = 0

    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            self.model.train() #  turn on batchnorm & dropout
            start = time.time()
            tr_avg_loss = self._run_one_epoch(epoch)
            val_loss, _continue = self._run_validation(epoch)
            if not _continue:
                break
            self._train_log('epoch', epoch=epoch, loss=tr_avg_loss, val_loss=val_loss,start=start)

            if self.checkpoint: #save model at each epoch
                file_path = os.path.join(self.save_folder, f'epoch{epoch+1}.pth.tar')
                torch.save({
                    'epoch' : epoch+1,
                    'model_state_dict' : self.model.state_dict(),
                    'optimizer_state' : self.optimizer.state_dict(),
                    'trandom_state' : torch.get_rng_state(),
                    'nrandom_state' : np.random.get_state()}, file_path)
                self.logger.info(f'Saving checkpoint model to {file_path}')

                # save the best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    best_file_path = os.path.join(self.save_folder, 'temp_best.pth.tar')
                    torch.save({
                        'epoch' : epoch+1,
                        'model_state_dict' : self.model.state_dict(),
                        'optimizer_state' : self.optimizer.state_dict(),
                        'trandom_state' : torch.get_rng_state(),
                        'nrandom_state' : np.random.get_state()}, best_file_path)
                    self.logger.info(f'Find better validated model, saving to {best_file_path}')

    def _run_one_epoch(self, epoch, validation=False):
        start = time.time()
        total_loss = 0
        data_loader = self.train_dataloader if not validation else self.val_dataloader
        # todo: add tensorboard loss tracking

        for i, (data) in enumerate(data_loader):
            noisy, clean = data
            if self.use_cuda:
                noisy, clean = noisy.cuda(), clean.cuda()
            estimated_clean = self.model(noisy)
            loss = self.criterion(clean, estimated_clean)
            if not validation:
                self.optimizer.zero_grad()
                loss.backward()
                if self.max_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
                self.optimizer.step()
            total_loss += loss.item()
            if (i % self.print_freq == 0) and (self.print_freq > 0):
                params =  {'it':i,'loss_item':loss.item()}
                self._train_log('freq', epoch, total_loss, start, **params)
            #todo: add tensorboard loss tracking here

        return total_loss/(i+1)

    def _run_validation(self, epoch):
        _continue=True
        self.model.eval() # turn off batchnorm & dropout
        with torch.no_grad():
            val_loss = self._run_one_epoch(epoch, validation=True)

        # adjust learning rate (halving)
        if self.half_lr:
            if val_loss >= self.prev_val_loss:
                self.val_no_impv += 1
                if self.val_no_impv >= 3:
                    self.halving = True
                if self.val_no_impv >= 10 and self.early_stop:
                    self.logger.info('No improvement for 10 epochs, early stopping.')
                    _continue=False
            else:
                self.val_no_impv = 0
        if self.halving:
            optim_state = self.optimizer.state_dict()
            optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr']/2
            self.optimizer.load_state_dict(optim_state)
            self.logger.info(f"Learning rate adjusted to {optim_state['param_groups'][0]['lr']:.6f}")
            self.halving = False
        self.prev_val_loss = val_loss
        #add tboard visualizing loss
        return val_loss, _continue

    def _train_log(self, _type, **kwargs):
        if _type == 'freq':
            # ARREGLAR!
            self.logger.info(
                    f'Epoch {epoch+1} | '
                    f'Iter {kwargs["it"]+1: >5} | '
                    f'Average Loss {(loss/(kwargs["it"]+1)):2.3f} | '
                    f'Current Loss {kwargs["loss_item"]:3.6f} | '
                    f'{(1000 * (time.time() - start) / (kwargs["it"] + 1)):2.1f} ms/batch')

        elif _type == 'epoch':
            self.logger.info(f"epoch {(kwargs['epoch']+1)} \tLoss {kwargs['loss']:5.3f} \tVal loss {kwargs['val_loss']:5.3f} \tdur {(time.time()-kwargs['start']):3.2f}s")
        elif _type == 'val_summary':
            pass
if __name__ == '__main__':
    print('testing')
