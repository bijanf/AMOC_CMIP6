"""Functions for processing CMIP6 netCDF files."""

import xarray as xr
import numpy as np
import pandas as pd
import gc
from ..config import LATITUDE_TARGET, DEPTH_MIN

def inspect_basin_structure(file_path):
    """Inspect the basin structure in a netCDF file.
    
    Args:
        file_path (str): Path to netCDF file
        
    Returns:
        dict: Information about basin structure
    """
    basin_info = {
        "has_basin_dim": False,
        "basin_values": None,
        "atlantic_basin_idx": None,
        "sector_values": None
    }
    
    try:
        with xr.open_dataset(file_path) as ds:
            if 'msftmz' not in ds:
                print("No msftmz variable found in the dataset")
                return basin_info
            
            msftmz = ds['msftmz']
            
            # Check if basin dimension exists
            if 'basin' in msftmz.dims:
                basin_info["has_basin_dim"] = True
                basin_info["basin_values"] = msftmz.basin.values.tolist()
                
                # Check for sector coordinate with basin names
                if 'sector' in ds:
                    basin_info["sector_values"] = [str(s) for s in ds['sector'].values]
                    
                    # Try to identify Atlantic basin
                    for i, sector in enumerate(ds['sector'].values):
                        sector_str = str(sector)
                        if 'atlantic' in sector_str.lower():
                            basin_info["atlantic_basin_idx"] = i
                            break
                
                # If Atlantic not found by name, check AMOC values
                if basin_info["atlantic_basin_idx"] is None:
                    lat_26 = msftmz.sel(lat=LATITUDE_TARGET, method='nearest')
                    max_amoc = -float('inf')
                    max_basin = None
                    
                    for basin_idx in basin_info["basin_values"]:
                        if basin_idx == 2 and "global" in str(ds.get('sector', [""])[2]).lower():
                            # Skip global basin
                            continue
                            
                        basin_data = lat_26.sel(basin=basin_idx)
                        max_val = float(basin_data.max())
                        
                        # Convert to Sv if needed
                        if 'kg' in msftmz.attrs.get('units', '').lower():
                            max_val /= 1e9
                            
                        if max_val > max_amoc:
                            max_amoc = max_val
                            max_basin = basin_idx
                    
                    if max_basin is not None and max_amoc > 10:
                        basin_info["atlantic_basin_idx"] = max_basin
    
    except Exception as e:
        print(f"Error inspecting file: {e}")
    
    return basin_info

def identify_atlantic_basin(file_path, model_name):
    """Identify the Atlantic basin index in a CMIP6 file.
    
    Args:
        file_path (str): Path to netCDF file
        model_name (str): Name of the model
        
    Returns:
        int: Index of the Atlantic basin, or None if not found
    """
    basin_info = inspect_basin_structure(file_path)
    
    # If Atlantic basin was identified in the inspection
    if basin_info["atlantic_basin_idx"] is not None:
        return basin_info["atlantic_basin_idx"]
    
    # Model-specific handling
    if model_name == "CanESM5":
        return 0  # We know basin 0 is Atlantic for CanESM5
    
    # Default to first basin if we can't identify Atlantic
    if basin_info["has_basin_dim"] and basin_info["basin_values"]:
        return basin_info["basin_values"][0]
    
    return None
