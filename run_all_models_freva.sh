#!/bin/bash
#SBATCH --job-name=amoc_freva
#SBATCH --output=amoc_freva_%j.out
#SBATCH --error=amoc_freva_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --account=kd1418
#SBATCH --partition=compute
#SBATCH --time=01:00:00
#SBATCH --mem=64G

# Load necessary modules
module load clint codes

# Run the Python script for all models using freva
python scripts/run_all_models_freva.py 