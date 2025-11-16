#!/usr/bin/env bash
        
#SBATCH --job-name=muzero_smt_train
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

#SBATCH --nodes=1    
#SBATCH --ntasks=1                     
#SBATCH --cpus-per-task=32
#SBATCH --mem=64000
#SBATCH --gpus=h100:1
#SBATCH --time=1-00:00:00     


source venv/bin/activate

python -u train.py