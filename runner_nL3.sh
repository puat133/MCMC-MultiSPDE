#!/usr/bin/bash
#SBATCH --time=5-0 #time-requested
#SBATCH --mem-per-cpu=12G # memory
#SBATCH --gres=gpu:v100:1
#SBATCH --output twoDtest2-%J.log
##First activate CUPYrc
module load cuda/10.1.243
source /home/${USER}/.bashrc
conda activate CUPYrc 

iter=$1
seq_no=$2
nFourier=$3 #First Argument is Fourier Basis
nExtended=$4 #Second Argument
nTheta=$5 #Third Arugment
nSamples=$6
sigmaV=$7
sigmaScalling=$8
initFolder=$9


python twoDsim.py --n-layers=3 --iter=$iter --seq-no=$seq_no --n=$nFourier --n-extended=$nExtended --n-theta=$nTheta --meas-type=tomo --meas-std=2e-1 --n-samples=$nSamples --phantom-name=shepp.png --sigma-v=$sigmaV --sigma-scaling=$sigmaScalling --init-folder=$initFolder --verbose