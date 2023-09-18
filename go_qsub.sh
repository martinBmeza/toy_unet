#!/bin/bash
#$ -S /bin/bash
#$ -N toy_unet
#$ -q short.q@@stable
#$ -l ram_free=32G,mem_free=32G
#$ -l matylda3=0.5
#$ -o /homes/eva/q/qmeza/rhome/toy_unet/logger/out.log
#$ -e /homes/eva/q/qmeza/rhome/toy_unet/logger/err.log


# -l gpu=1,gpu_ram=8G
source ~qmeza/.bashrc
conda activate torch
#export CUDA_VISIBLE_DEVICES=$(free-gpus.sh 1)
home="/homes/eva/q/qmeza/rhome/"
cd $home
python3 test_gpu.py
root_dir=$home"toy_unet"
cd $root_dir
echo "-------- Started --------"
bash setup.sh
echo "-------- END --------"
