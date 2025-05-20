#!/bin/bash
#SBATCH --job-name=amoc_analysis
#SBATCH --output=amoc_analysis_%j.out
#SBATCH --error=amoc_analysis_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --account=kd1418
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --mem=32G

# Load necessary modules
module load clint codes

# Set model name (default to CanESM5 if not provided)
MODEL=${1:-CanESM5}

# Run the Python script
python scripts/run_multi_scenario.py $MODEL