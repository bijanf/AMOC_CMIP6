#!/bin/bash
#SBATCH --job-name=amoc_cmip5_cmip6
#SBATCH --output=amoc_cmip5_cmip6_%j.out
#SBATCH --error=amoc_cmip5_cmip6_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --account=kd1418
#SBATCH --partition=compute
#SBATCH --time=02:00:00
#SBATCH --mem=64G

# Load necessary modules
module load clint codes

# Run the Python script for both CMIP5 and CMIP6 models, limiting plots to 2100
python scripts/run_all_models_freva.py --all --cmip6 --end-year 2100 