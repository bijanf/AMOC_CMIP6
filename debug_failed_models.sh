#!/bin/bash
#SBATCH --job-name=debug_amoc
#SBATCH --output=debug_amoc_%j.out
#SBATCH --error=debug_amoc_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --account=kd1418
#SBATCH --partition=compute
#SBATCH --time=02:00:00
#SBATCH --mem=32G

# Load necessary modules
module load clint codes

# Run the Python script to debug failed models
python scripts/debug_failed_models.py 