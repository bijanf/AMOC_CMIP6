"""Configuration settings for CMIP6 AMOC analysis."""

import os

# Base path for CMIP6 data
CMIP6_DATA_PATH = "/work/ik1017/CMIP6/data/CMIP6"

# AMOC analysis settings
LATITUDE_TARGET = 26.5  # RAPID array latitude
DEPTH_MIN = 500  # Minimum depth (m) for overturning cell

# Available scenarios
SCENARIOS = {
    "historical": "Historical",
    "ssp119": "SSP1-1.9",
    "ssp126": "SSP1-2.6",
    "ssp245": "SSP2-4.5",
    "ssp370": "SSP3-7.0",
    "ssp434": "SSP4-3.4",
    "ssp460": "SSP4-6.0",
    "ssp534-over": "SSP5-3.4-OS",
    "ssp585": "SSP5-8.5",
    "rcp26": "RCP2.6",
    "rcp45": "RCP4.5",
    "rcp60": "RCP6.0",
    "rcp85": "RCP8.5"
}

# Known models with good AMOC representation
RECOMMENDED_MODELS = [
    "CanESM5", "MIROC6", "MPI-ESM1-2-LR", "NorESM2-LM", 
    "UKESM1-0-LL", "CESM2", "GFDL-ESM4", "EC-Earth3"
]

# Define the directory for plots
PLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots")
