#!/bin/bash

#SBATCH --time=3-00:00:00
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --mem 32gb
#SBATCH --gres=gpu:gtx1080ti:1
#SBATCH --error error/CGI_128.out
#SBATCH --output output/CGI_128.out
#SBATCH --job-name=CGI_128

source /cfarhomes/hzhou/.bashrc
source /fs/vulcan-scratch/hzhou/pytorch/bin/activate

python trainCGI_fine.py result/result_CGI/result_fine_lessData 1e-3 0 10 result/result_CGI/result_coarse_lessData_continue_0.0010_0.00_0100
