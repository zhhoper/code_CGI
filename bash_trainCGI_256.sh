#!/bin/bash

#SBATCH --time=3-00:00:00
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --mem 32gb
#SBATCH --gres=gpu:gtx1080ti:1
#SBATCH --error error/CGI_256_noflip.out
#SBATCH --output output/CGI_256_noflip.out
#SBATCH --job-name=CGI_256_noflip

source /cfarhomes/hzhou/.bashrc
source /fs/vulcan-scratch/hzhou/pytorch/bin/activate

python trainCGI.py defineHourglass_CGI result/result_CGI/result_256_noflip 256 300 --lr 1e-3 --epoch 30 --wd 0
