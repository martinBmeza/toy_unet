#!/usr/bin/bash
set -x

python train.py \
	--name='test_experiment'\
	--train_data_dir='data/train'\
	--val_data_dir='data/val'\
	--use_cuda=1\
	--epochs=200\
	--half_lr=0\
	--early_stop=1\
	--max_norm=500\
	--shuffle=1\
	--batch_size=256\
	--num_workers=8\
	--optimizer='adam'\
        --loss='mse'\
	--lr=0.01\
	--momentum=0.0\
	--l2=0.0\
	--save_folder='experiments'\
	--checkpoint=0\
	--continue_from=''\
        --model_path='final.pth.tar'\
	--print_freq=-1\
	--tboard_epoch=1\
	--tboard_graph=1\

