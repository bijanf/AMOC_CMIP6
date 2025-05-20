#!/bin/bash
#SBATCH --job-name=amoc_all_models
#SBATCH --output=amoc_all_models_%j.out
#SBATCH --error=amoc_all_models_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --account=kd1418
#SBATCH --partition=compute
#SBATCH --time=2:00:00
#SBATCH --mem=64G

# Load necessary modules
module load clint codes

# Run the Python script for all models
python scripts/run_all_models.py
