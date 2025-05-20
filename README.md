# AMOC Analysis Project

This repository contains code for analyzing the Atlantic Meridional Overturning Circulation (AMOC) using CMIP5 and CMIP6 climate model data.

## Project Overview

This project analyzes AMOC data from various climate models participating in CMIP5 and CMIP6. The analysis includes processing, visualization, and statistical analysis of AMOC data to understand its behavior and changes across different climate models.

## Directory Structure

- `scripts/`: Contains Python scripts for data processing and analysis
- `plots/`: Directory for storing generated plots and visualizations
- `output/`: Directory for storing processed data and analysis results
- `cmip6_amoc/`: Contains CMIP6-specific analysis code and data

## Requirements

The project requires the following Python packages:
- numpy >= 1.20.0
- pandas >= 1.3.0
- xarray >= 0.20.0
- matplotlib >= 3.4.0
- netCDF4 >= 1.5.7
- cftime >= 1.5.0
- seaborn >= 0.11.0

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

The main analysis can be run using the provided shell script:

```bash
./run_cmip5_and_cmip6.sh
```

This script will:
1. Process CMIP5 and CMIP6 model data
2. Generate AMOC analysis
3. Create visualizations
4. Store results in the output directory

## Additional Scripts

- `run_all_cmip6_models.sh`: Run analysis for all CMIP6 models
- `debug_specific_model.sh`: Debug analysis for a specific model
- `debug_failed_models.sh`: Debug analysis for failed model runs
- `run_amoc.sh`: Run AMOC analysis for individual models

## Output

The analysis generates:
- Processed AMOC data files
- Statistical analysis results
- Visualization plots
- Error logs and output files

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Specify your license here]

## Contact

[Your contact information] 