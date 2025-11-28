#!/usr/bin/env bash
        
#SBATCH --job-name=smt_test
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

#SBATCH --exclusive
#SBATCH --nodes=1    
#SBATCH --ntasks=1                     
#SBATCH --cpus-per-task=192
#sbatch --mem=0
#SBATCH --time=1-00:00:00

source venv/bin/activate

python -u test.py "$@"