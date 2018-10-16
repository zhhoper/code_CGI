#!/bin/bash

#SBATCH --time=3-00:00:00
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --mem 32gb
#SBATCH --gres=gpu:gtx1080ti:1
#SBATCH --error error/lessData_continue.out
#SBATCH --output output/lessData_continue.out
#SBATCH --job-name=lessData_continue

source /cfarhomes/hzhou/.bashrc
source /fs/vulcan-scratch/hzhou/pytorch/bin/activate

python trainCGI_coarse_lessData_continue.py result/result_CGI/result_coarse_lessData_continue 1e-3 0 10
