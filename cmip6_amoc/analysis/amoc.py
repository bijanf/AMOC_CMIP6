"""Functions for calculating AMOC indices."""

import xarray as xr
import numpy as np
import pandas as pd
import gc
from ..config import LATITUDE_TARGET, DEPTH_MIN
from ..data.finder import extract_ensemble_member
from ..data.processor import identify_atlantic_basin

def compute_amoc_index(files, scenario, model_name):
    """Compute AMOC index from a list of files.
    
    Args:
        files (list): List of file paths
        scenario (str): Scenario name
        model_name (str): Model name
        
    Returns:
        pd.DataFrame: DataFrame with years as index and ensemble members as columns
    """
    if not files:
        print(f"No files for {scenario}")
        return None
        
    print(f"Processing {scenario} files for {model_name}...")
    
    # Dictionary to store AMOC values by ensemble member
    ensemble_data = {}
    
    # Identify Atlantic basin from the first file
    atlantic_basin = identify_atlantic_basin(files[0], model_name)
    if atlantic_basin is not None:
        print(f"Using basin={atlantic_basin} as Atlantic for {model_name}")
    
    for file in files:
        try:
            print(f"Processing {file}")
            # Extract ensemble member from filename
            ensemble = extract_ensemble_member(file)
            
            # Open dataset with cftime enabled to avoid datetime64 warnings
            with xr.open_dataset(file, use_cftime=True) as ds:
                # Get msftmz and convert to Sv if necessary
                psi = ds['msftmz']
                if 'kg' in psi.attrs.get('units', '').lower():
                    psi_sv = psi / 1e9
                    print(f"Converting units from kg/s to Sv")
                else:
                    psi_sv = psi
                
                # Select Atlantic basin if needed
                if 'basin' in psi_sv.dims and atlantic_basin is not None:
                    psi_sv = psi_sv.sel(basin=atlantic_basin)
                
                # Select latitude closest to target
                psi_26 = psi_sv.sel(lat=LATITUDE_TARGET, method='nearest')
                actual_lat = float(psi_26.lat.values)
                print(f"Using latitude {actual_lat} (closest to target {LATITUDE_TARGET})")
                
                # Filter to depths below minimum
                psi_26_deep = psi_26.sel(lev=slice(DEPTH_MIN, None))
                
                # Get max overturning at this latitude
                amoc_index = psi_26_deep.max(dim='lev')
                
                # Check if AMOC values seem reasonable
                mean_amoc = float(amoc_index.mean())
                if mean_amoc < 10:
                    print(f"WARNING: Mean AMOC value ({mean_amoc:.2f} Sv) is unusually low!")
                elif mean_amoc > 30:
                    print(f"WARNING: Mean AMOC value ({mean_amoc:.2f} Sv) is unusually high!")
                else:
                    print(f"Mean AMOC for {ensemble}: {mean_amoc:.2f} Sv")
                
                # Extract years and values
                years = np.array([t.year for t in amoc_index.time.values])
                values = amoc_index.values
                
                # Store data by ensemble member
                for year, value in zip(years, values):
                    if ensemble not in ensemble_data:
                        ensemble_data[ensemble] = {}
                    
                    if year in ensemble_data[ensemble]:
                        # Average if we have multiple values for the same year
                        ensemble_data[ensemble][year] = (ensemble_data[ensemble][year] + value) / 2
                    else:
                        ensemble_data[ensemble][year] = value
                
                # Force garbage collection
                gc.collect()
                
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    if not ensemble_data:
        print(f"No valid data could be processed for {scenario}")
        return None
    
    # Convert to DataFrame with years as index and ensemble members as columns
    all_years = sorted(set(year for member_data in ensemble_data.values() for year in member_data.keys()))
    df = pd.DataFrame(index=all_years)
    
    for ensemble, years_data in ensemble_data.items():
        df[ensemble] = pd.Series(years_data)
    
    return df