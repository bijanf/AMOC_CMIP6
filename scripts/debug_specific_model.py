#!/usr/bin/env python3
"""
Script to debug a specific CMIP6 model in detail.

Usage:
    python debug_specific_model.py MODEL_NAME
"""

import sys
import os
import pandas as pd
import traceback
import argparse
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cmip6_amoc.config import SCENARIOS, LATITUDE_TARGET, DEPTH_MIN
from cmip6_amoc.data.freva_finder import find_files_freva
from cmip6_amoc.data.processor import identify_atlantic_basin

# All SSP scenarios to check
ALL_SSP_SCENARIOS = ["ssp119", "ssp126", "ssp245", "ssp370", "ssp434", "ssp460", "ssp534-over", "ssp585"]

def inspect_file(file_path, model_name):
    """Inspect a netCDF file in detail."""
    print(f"\nInspecting file: {file_path}")
    
    try:
        # Open the dataset
        with xr.open_dataset(file_path, use_cftime=True) as ds:
            # Print basic information
            print(f"Dataset dimensions: {ds.dims}")
            print(f"Dataset variables: {list(ds.data_vars)}")
            print(f"Dataset coordinates: {list(ds.coords)}")
            
            # Check if msftmz exists
            if 'msftmz' not in ds:
                print("ERROR: msftmz variable not found in dataset")
                return
            
            # Get msftmz variable
            msftmz = ds['msftmz']
            print(f"msftmz shape: {msftmz.shape}")
            print(f"msftmz dimensions: {msftmz.dims}")
            print(f"msftmz attributes: {msftmz.attrs}")
            
            # Check units
            units = msftmz.attrs.get('units', 'unknown')
            print(f"msftmz units: {units}")
            
            # Check if basin dimension exists
            if 'basin' in msftmz.dims:
                basins = ds.basin.values
                print(f"Basin values: {basins}")
                
                # Try to identify Atlantic basin
                atlantic_basin = identify_atlantic_basin(file_path, model_name)
                print(f"Identified Atlantic basin: {atlantic_basin}")
                
                # Select Atlantic basin if possible
                if atlantic_basin is not None:
                    msftmz_atlantic = msftmz.sel(basin=atlantic_basin)
                    print(f"msftmz_atlantic shape: {msftmz_atlantic.shape}")
                else:
                    print("WARNING: Could not identify Atlantic basin")
                    msftmz_atlantic = msftmz
            else:
                print("No basin dimension found")
                msftmz_atlantic = msftmz
            
            # Check latitude values
            if 'lat' in msftmz.dims:
                lats = msftmz.lat.values
                print(f"Latitude values: {lats}")
                
                # Find closest latitude to target
                lat_idx = np.abs(lats - LATITUDE_TARGET).argmin()
                closest_lat = lats[lat_idx]
                print(f"Closest latitude to {LATITUDE_TARGET}N: {closest_lat}N")
                
                # Select data at this latitude
                msftmz_lat = msftmz_atlantic.sel(lat=closest_lat, method='nearest')
                print(f"msftmz_lat shape: {msftmz_lat.shape}")
            else:
                print("ERROR: No latitude dimension found")
                return
            
            # Check depth values
            if 'lev' in msftmz.dims:
                levs = msftmz.lev.values
                print(f"Depth levels: {len(levs)} levels from {levs.min()} to {levs.max()}")
                
                # Filter to depths below minimum
                msftmz_deep = msftmz_lat.sel(lev=slice(DEPTH_MIN, None))
                print(f"msftmz_deep shape: {msftmz_deep.shape}")
            else:
                print("ERROR: No depth dimension found")
                return
            
            # Calculate AMOC index
            amoc_index = msftmz_deep.max(dim='lev')
            print(f"AMOC index shape: {amoc_index.shape}")
            
            # Calculate basic statistics
            mean_amoc = float(amoc_index.mean())
            min_amoc = float(amoc_index.min())
            max_amoc = float(amoc_index.max())
            print(f"AMOC index statistics: mean={mean_amoc:.2f}, min={min_amoc:.2f}, max={max_amoc:.2f}")
            
            # Check if values seem reasonable
            if mean_amoc < 5:
                print("WARNING: Mean AMOC value is unusually low!")
            elif mean_amoc > 30:
                print("WARNING: Mean AMOC value is unusually high!")
            
            # Create a quick plot
            plt.figure(figsize=(10, 6))
            amoc_index.plot()
            plt.title(f"AMOC Index at {closest_lat}N for {model_name}")
            plt.ylabel("AMOC Strength (Sv)")
            plt.grid(True, alpha=0.3)
            
            # Save the plot
            plot_filename = f"debug_{model_name}_amoc_index.png"
            plt.savefig(plot_filename, dpi=300)
            print(f"Plot saved as: {plot_filename}")
            
            # Create a depth-time plot
            plt.figure(figsize=(12, 8))
            msftmz_lat.plot(x='time', y='lev', yincrease=False, cmap='RdBu_r')
            plt.title(f"AMOC Streamfunction at {closest_lat}N for {model_name}")
            plt.ylabel("Depth (m)")
            plt.xlabel("Time")
            
            # Save the plot
            plot_filename = f"debug_{model_name}_amoc_depth_time.png"
            plt.savefig(plot_filename, dpi=300)
            print(f"Plot saved as: {plot_filename}")
            
    except Exception as e:
        print(f"ERROR inspecting file: {e}")
        traceback.print_exc()

def find_available_scenarios(model_name):
    """Find all available scenarios for a model.
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        dict: Dictionary mapping scenario names to file lists
    """
    scenario_files = {}
    
    # Always check historical
    historical_files = find_files_freva(model_name, "historical")
    if historical_files:
        scenario_files["historical"] = historical_files
    
    # Check all SSP scenarios
    for scenario in ALL_SSP_SCENARIOS:
        files = find_files_freva(model_name, scenario)
        if files:
            scenario_files[scenario] = files
    
    return scenario_files

def main():
    """Main function to debug a specific model."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Debug a specific CMIP6 model')
    parser.add_argument('model', help='Name of the CMIP6 model to debug')
    args = parser.parse_args()
    
    model_name = args.model
    print(f"Debugging model: {model_name}")
    
    # Find all available scenarios for this model
    print(f"\nFinding available scenarios for model {model_name}...")
    scenario_files = find_available_scenarios(model_name)
    
    if not scenario_files:
        print(f"No data found for any scenario for model {model_name}")
        return
    
    print(f"\nFound data for {len(scenario_files)} scenarios:")
    for scenario, files in scenario_files.items():
        print(f"  - {scenario}: {len(files)} files")
    
    # Inspect one file from each scenario
    for scenario, files in scenario_files.items():
        print(f"\n{'='*50}")
        print(f"Inspecting {scenario} data for {model_name}")
        print(f"{'='*50}")
        
        # Inspect the first file
        inspect_file(files[0], model_name)

if __name__ == "__main__":
    main() 