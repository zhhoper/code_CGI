#!/bin/bash

#SBATCH --time=3-00:00:00
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --mem 32gb
#SBATCH --gres=gpu:gtx1080ti:1
#SBATCH --error error/CGI_withLighting_256.out
#SBATCH --output output/CGI_withLighting_256.out
#SBATCH --job-name=CGI_wl256

source /cfarhomes/hzhou/.bashrc
source /fs/vulcan-scratch/hzhou/pytorch/bin/activate

python trainCGI_withLighting.py defineHourglass_128 result/result_CGI/result_withLighting_256 256 300 --lr 1e-3 --epoch 30 --wd 0
