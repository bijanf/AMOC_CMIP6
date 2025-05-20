#!/bin/bash
#SBATCH --job-name=amoc_all_cmip6
#SBATCH --output=amoc_all_cmip6_%j.out
#SBATCH --error=amoc_all_cmip6_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --account=kd1418
#SBATCH --partition=compute
#SBATCH --time=02:00:00
#SBATCH --mem=64G

# Load necessary modules
# module load clint codes
module load clint freva

# Run the Python script for all CMIP6 models
python scripts/run_all_models_freva.py --cmip6 --end-year 2100 --all