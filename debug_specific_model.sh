#!/bin/bash
#SBATCH --job-name=debug_model
#SBATCH --output=debug_model_%j.out
#SBATCH --error=debug_model_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=kd1418
#SBATCH --partition=compute
#SBATCH --time=01:00:00
#SBATCH --mem=16G

# Load necessary modules
module load clint codes

# Check if model name is provided
if [ $# -eq 0 ]; then
    echo "Error: No model name provided"
    echo "Usage: sbatch debug_specific_model.sh MODEL_NAME"
    exit 1
fi

# Run the Python script to debug a specific model
python scripts/debug_specific_model.py $1 