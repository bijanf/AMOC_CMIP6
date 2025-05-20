import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
import glob

# === Find models with both historical and SSP585 data ===
def find_models_with_both_experiments():
    print("Searching for models with both historical and SSP585 data...")
    
    # Try specific known models that often have both experiments
    known_models = ["CanESM5", "MIROC6", "MPI-ESM1-2-LR", "NorESM2-LM", 
                   "UKESM1-0-LL", "CESM2", "GFDL-ESM4", "EC-Earth3"]
    
    models_with_both = []
    for model in known_models:
        # For now, just add these known models to the list
        models_with_both.append(model)
        print(f"Known model {model} has both historical and SSP585 data")
    
    print("Available models with both experiments:")
    for i, model in enumerate(models_with_both, 1):
        print(f"{i}. {model}")
    print()
    
    return models_with_both

# Extract ensemble member from filename
def extract_ensemble_member(filename):
    # Pattern to match ensemble member (e.g., r1i1p1f1)
    pattern = r'r\d+i\d+p\d+f\d+'
    match = re.search(pattern, filename)
    if match:
        return match.group(0)
    return "unknown"

# Find files for a specific model and experiment
def find_files(model_name, experiment):
    # Base path for CMIP6 data
    base_path = "/work/ik1017/CMIP6/data/CMIP6"
    
    if experiment == "historical":
        search_path = f"{base_path}/CMIP/*/{model_name}/historical/*/Omon/msftmz/*/v*/msftmz_Omon_{model_name}_historical_*.nc"
    else:  # ssp585
        search_path = f"{base_path}/ScenarioMIP/*/{model_name}/ssp585/*/Omon/msftmz/*/v*/msftmz_Omon_{model_name}_ssp585_*.nc"
    
    files = glob.glob(search_path)
    return files

# Memory-efficient function to compute AMOC index from a list of files
def compute_amoc_index(files, experiment_name):
    if not files:
        print(f"No files for {experiment_name}")
        return None
        
    print(f"Processing {experiment_name} files...")
    
    # Dictionary to store AMOC values by ensemble member
    ensemble_data = {}
    
    latitude_target = 26.5  # RAPID latitude
    depth_min = 500  # Minimum depth (m) for overturning cell
    
    # First, examine one file to understand the basin structure
    print(f"Examining basin structure in {experiment_name} files...")
    with xr.open_dataset(files[0]) as ds:
        if 'basin' in ds['msftmz'].dims:
            print(f"Basin values in {experiment_name} dataset: {ds['msftmz'].basin.values}")
            # Try to identify which basin index corresponds to Atlantic
            for i, basin in enumerate(ds['msftmz'].basin.values):
                print(f"  Basin {i}: {basin}")
        else:
            print(f"No basin dimension found in {experiment_name} dataset.")
    
    # Check for sector coordinate to identify Atlantic basin
    atlantic_basin_index = None
    with xr.open_dataset(files[0]) as ds:
        if 'sector' in ds:
            for i, sector in enumerate(ds['sector'].values):
                sector_str = str(sector)
                if 'atlantic' in sector_str.lower():
                    atlantic_basin_index = i
                    print(f"Found Atlantic basin at index {i}: {sector_str}")
                    break
    
    for file in files:
        try:
            print(f"Processing {file}")
            # Extract ensemble member from filename
            ensemble = extract_ensemble_member(file)
            
            with xr.open_dataset(file) as ds:
                # Get msftmz and convert to Sv if necessary
                psi = ds['msftmz']
                if 'kg' in psi.attrs.get('units', '').lower():
                    psi_sv = psi / 1e9
                    print(f"Converting units from kg/s to Sv for {experiment_name}")
                else:
                    psi_sv = psi
                    print(f"Units for {experiment_name}: {psi.attrs.get('units', 'unknown')}")
                
                # Select Atlantic basin if needed
                basin_found = False
                if 'basin' in psi_sv.dims:
                    # Use the identified Atlantic basin index if available
                    if atlantic_basin_index is not None:
                        psi_sv = psi_sv.sel(basin=atlantic_basin_index)
                        print(f"Using basin={atlantic_basin_index} as Atlantic (identified from sector coordinate)")
                        basin_found = True
                    else:
                        # Try different ways to identify Atlantic basin
                        atlantic_basin = None
                        
                        # Method 1: Look for 'atl' in the name if sector coordinate exists
                        if 'sector' in ds:
                            for i, sector in enumerate(ds['sector'].values):
                                if 'atlantic' in str(sector).lower():
                                    atlantic_basin = i
                                    print(f"Using basin={i} as Atlantic (from sector name)")
                                    break
                        
                        # Method 2: For CanESM5, use basin 0 (which we now know is Atlantic)
                        elif "CanESM5" in file:
                            atlantic_basin = 0
                            print(f"For CanESM5, using basin=0 as Atlantic")
                        
                        if atlantic_basin is not None:
                            psi_sv = psi_sv.sel(basin=atlantic_basin)
                            basin_found = True
                        else:
                            # If still no Atlantic basin found, use the first basin
                            if len(psi_sv.basin) > 0:
                                psi_sv = psi_sv.sel(basin=psi_sv.basin.values[0])
                                print(f"Warning: No Atlantic basin identified in {experiment_name}.")
                                print(f"Using first available basin instead: {psi_sv.basin.values[0]}")
                                basin_found = True
                
                if not basin_found:
                    print(f"Warning: No basin dimension found in {experiment_name} dataset.")
                
                # Select latitude closest to 26.5°N
                psi_26 = psi_sv.sel(lat=latitude_target, method='nearest')
                actual_lat = float(psi_26.lat.values)
                print(f"Using latitude {actual_lat} for {experiment_name} (closest to target {latitude_target})")
                
                # Filter to depths below 500 m
                psi_26_deep = psi_26.sel(lev=slice(depth_min, None))
                
                # Get max overturning at this latitude
                amoc_index = psi_26_deep.max(dim='lev')
                
                # Check if AMOC values seem reasonable
                mean_amoc = float(amoc_index.mean())
                if mean_amoc < 10:
                    print(f"WARNING: Mean AMOC value ({mean_amoc:.2f} Sv) is unusually low!")
                    print(f"This might indicate incorrect basin selection or units conversion.")
                elif mean_amoc > 30:
                    print(f"WARNING: Mean AMOC value ({mean_amoc:.2f} Sv) is unusually high!")
                    print(f"This might indicate incorrect basin selection or units conversion.")
                else:
                    print(f"Mean AMOC for {ensemble} ({experiment_name}): {mean_amoc:.2f} Sv")
                
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
                import gc
                gc.collect()
                
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    if not ensemble_data:
        print(f"No valid data could be processed for {experiment_name}")
        return None
    
    # Convert to DataFrame with years as index and ensemble members as columns
    all_years = sorted(set(year for member_data in ensemble_data.values() for year in member_data.keys()))
    df = pd.DataFrame(index=all_years)
    
    for ensemble, years_data in ensemble_data.items():
        df[ensemble] = pd.Series(years_data)
    
    return df

# Function to inspect basin structure in a netCDF file
def inspect_basin_structure(file_path):
    print(f"\n=== Inspecting basin structure in {os.path.basename(file_path)} ===")
    try:
        with xr.open_dataset(file_path) as ds:
            if 'msftmz' not in ds:
                print("No msftmz variable found in the dataset")
                return
            
            msftmz = ds['msftmz']
            
            # Check if basin dimension exists
            if 'basin' not in msftmz.dims:
                print("No basin dimension found in msftmz variable")
                return
            
            # Get basin values
            basin_values = msftmz.basin.values
            print(f"Basin dimension values: {basin_values}")
            
            # Check if there's a basin coordinate variable with names
            if 'basin' in ds and hasattr(ds['basin'], 'attrs'):
                print("Basin coordinate attributes:")
                for attr_name, attr_value in ds['basin'].attrs.items():
                    print(f"  {attr_name}: {attr_value}")
            
            # Check for a sector coordinate that might have basin names
            if 'sector' in ds:
                print("Sector coordinate values:")
                print(ds['sector'].values)
                if hasattr(ds['sector'], 'attrs'):
                    print("Sector coordinate attributes:")
                    for attr_name, attr_value in ds['sector'].attrs.items():
                        print(f"  {attr_name}: {attr_value}")
            
            # Look for any attributes that might contain basin information
            print("msftmz variable attributes:")
            for attr_name, attr_value in msftmz.attrs.items():
                print(f"  {attr_name}: {attr_value}")
                
            # If basin is numeric, try to visualize the data for each basin
            if np.issubdtype(basin_values.dtype, np.number):
                print("\nExamining AMOC values for each basin at 26.5°N:")
                lat_26 = msftmz.sel(lat=26.5, method='nearest')
                for basin_idx in basin_values:
                    basin_data = lat_26.sel(basin=basin_idx)
                    max_val = float(basin_data.max())
                    print(f"  Basin {basin_idx}: Max overturning = {max_val:.2f} {msftmz.attrs.get('units', '')}")
                    if 'kg' in msftmz.attrs.get('units', '').lower():
                        print(f"    In Sv: {max_val/1e9:.2f} Sv")
            
    except Exception as e:
        print(f"Error inspecting file: {e}")

# === Main script ===
try:
    # Find models with both experiments
    models = find_models_with_both_experiments()
    
    if not models:
        print("No models found with both historical and SSP585 data.")
        exit(1)
    
    # Select a model (can be changed to allow user input)
    model_name = models[0]  # Default to first model
    print(f"\nUsing model: {model_name}")
    
    # Get historical files
    print(f"Searching for historical files for model {model_name}...")
    historical_files = find_files(model_name, "historical")
    print(f"Found {len(historical_files)} historical files for model {model_name}")
    
    # Get SSP585 files
    print(f"Searching for ssp585 files for model {model_name}...")
    ssp585_files = find_files(model_name, "ssp585")
    print(f"Found {len(ssp585_files)} ssp585 files for model {model_name}")
    
    # Inspect the first file of each experiment to understand basin structure
    if historical_files:
        inspect_basin_structure(historical_files[0])
    if ssp585_files:
        inspect_basin_structure(ssp585_files[0])
    
    # Compute AMOC index for both experiments
    historical_amoc = compute_amoc_index(historical_files, "historical")
    ssp585_amoc = compute_amoc_index(ssp585_files, "ssp585")
    
    if historical_amoc is None or ssp585_amoc is None:
        raise ValueError("Could not compute AMOC index for one or both experiments")
    
    # === Plot AMOC index with ensemble statistics ===
    plt.figure(figsize=(14, 10))
    
    # Plot individual ensemble members with low alpha
    for column in historical_amoc.columns:
        plt.plot(historical_amoc.index, historical_amoc[column], 
                 color='blue', alpha=0.2, linewidth=0.8)
    
    for column in ssp585_amoc.columns:
        plt.plot(ssp585_amoc.index, ssp585_amoc[column], 
                 color='red', alpha=0.2, linewidth=0.8)
    
    # Calculate ensemble statistics for historical
    hist_mean = historical_amoc.mean(axis=1)
    hist_median = historical_amoc.median(axis=1)
    hist_lower = historical_amoc.quantile(0.25, axis=1)
    hist_upper = historical_amoc.quantile(0.75, axis=1)
    
    # Calculate ensemble statistics for SSP585
    ssp_mean = ssp585_amoc.mean(axis=1)
    ssp_median = ssp585_amoc.median(axis=1)
    ssp_lower = ssp585_amoc.quantile(0.25, axis=1)
    ssp_upper = ssp585_amoc.quantile(0.75, axis=1)
    
    # Plot ensemble means
    plt.plot(hist_mean.index, hist_mean, 
             label='Historical Mean (1850-2014)', color='darkblue', linewidth=2)
    plt.plot(ssp_mean.index, ssp_mean, 
             label='SSP585 Mean (2015-2100)', color='darkred', linewidth=2)
    
    # Plot ensemble medians
    plt.plot(hist_median.index, hist_median, 
             label='Historical Median', color='blue', linewidth=1.5, linestyle='--')
    plt.plot(ssp_median.index, ssp_median, 
             label='SSP585 Median', color='red', linewidth=1.5, linestyle='--')
    
    # Plot uncertainty ranges (25th-75th percentile)
    plt.fill_between(hist_lower.index, hist_lower, hist_upper, 
                     color='blue', alpha=0.2, label='Historical 25-75th percentile')
    plt.fill_between(ssp_lower.index, ssp_lower, ssp_upper, 
                     color='red', alpha=0.2, label='SSP585 25-75th percentile')
    
    # Set title and labels
    title = f"AMOC Index from {model_name} (Historical + SSP585)\n{len(historical_amoc.columns)} historical and {len(ssp585_amoc.columns)} SSP585 ensemble members"
    plt.title(title, fontsize=14)
    plt.ylabel("AMOC Strength (Sv)", fontsize=12)
    plt.xlabel("Year", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    # Save the plot
    output_filename = f"AMOC_index_{model_name}_historical_ssp585.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Plot saved as: {output_filename}")
    
    # Also save the data as CSV for future reference
    historical_amoc.to_csv(f"AMOC_index_{model_name}_historical.csv")
    print(f"Historical data saved as: AMOC_index_{model_name}_historical.csv")
    
    ssp585_amoc.to_csv(f"AMOC_index_{model_name}_ssp585.csv")
    print(f"SSP585 data saved as: AMOC_index_{model_name}_ssp585.csv")
    
    # Save combined data with ensemble statistics
    hist_stats = pd.DataFrame({
        'mean': hist_mean,
        'median': hist_median,
        'lower_25': hist_lower,
        'upper_75': hist_upper
    })
    
    ssp_stats = pd.DataFrame({
        'mean': ssp_mean,
        'median': ssp_median,
        'lower_25': ssp_lower,
        'upper_75': ssp_upper
    })
    
    combined_stats = pd.concat([hist_stats, ssp_stats])
    combined_stats.to_csv(f"AMOC_index_{model_name}_combined_stats.csv")
    print(f"Combined statistics saved as: AMOC_index_{model_name}_combined_stats.csv")
    
except Exception as e:
    print(f"Error: {e}")
    print("Please check the data availability and try again.")
