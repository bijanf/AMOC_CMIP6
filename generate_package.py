#!/usr/bin/env python3
"""
Script to generate the CMIP6 AMOC analysis package structure.
"""

import os
import sys

# Define the package structure and file contents
package_structure = {
    "cmip6_amoc/__init__.py": '''"""
CMIP6 AMOC Analysis Package

A package for analyzing AMOC (Atlantic Meridional Overturning Circulation)
in CMIP6 climate models.
"""

__version__ = '0.1.0'
''',
    "cmip6_amoc/config.py": '''"""Configuration settings for CMIP6 AMOC analysis."""

# Base path for CMIP6 data
CMIP6_DATA_PATH = "/work/ik1017/CMIP6/data/CMIP6"

# AMOC analysis settings
LATITUDE_TARGET = 26.5  # RAPID array latitude
DEPTH_MIN = 500  # Minimum depth (m) for overturning cell

# Available scenarios
SCENARIOS = {
    "historical": {"path_component": "CMIP", "color": "navy", "linestyle": "-"},
    "ssp126": {"path_component": "ScenarioMIP", "color": "green", "linestyle": "-"},
    "ssp245": {"path_component": "ScenarioMIP", "color": "gold", "linestyle": "-"},
    "ssp370": {"path_component": "ScenarioMIP", "color": "orange", "linestyle": "-"},
    "ssp585": {"path_component": "ScenarioMIP", "color": "darkred", "linestyle": "-"}
}

# Known models with good AMOC representation
RECOMMENDED_MODELS = [
    "CanESM5", "MIROC6", "MPI-ESM1-2-LR", "NorESM2-LM", 
    "UKESM1-0-LL", "CESM2", "GFDL-ESM4", "EC-Earth3"
]
''',
    "cmip6_amoc/data/__init__.py": '"""Data handling module for CMIP6 AMOC analysis."""',
    "cmip6_amoc/data/finder.py": '''"""Functions for finding CMIP6 data files."""

import os
import glob
import re
from ..config import CMIP6_DATA_PATH, SCENARIOS

def extract_ensemble_member(filename):
    """Extract ensemble member ID from filename.
    
    Args:
        filename (str): Path to CMIP6 netCDF file
        
    Returns:
        str: Ensemble member ID (e.g., r1i1p1f1)
    """
    pattern = r'r\d+i\d+p\d+f\d+'
    match = re.search(pattern, filename)
    if match:
        return match.group(0)
    return "unknown"

def find_models_with_scenarios(scenarios=None):
    """Find models that have data for all specified scenarios.
    
    Args:
        scenarios (list): List of scenario names to check
        
    Returns:
        list: Models with data for all specified scenarios
    """
    if scenarios is None:
        scenarios = ["historical", "ssp585"]
    
    from ..config import RECOMMENDED_MODELS
    
    models_with_all = []
    for model in RECOMMENDED_MODELS:
        has_all = True
        for scenario in scenarios:
            files = find_files(model, scenario)
            if not files:
                has_all = False
                break
        
        if has_all:
            models_with_all.append(model)
            print(f"Model {model} has data for all specified scenarios")
    
    return models_with_all

def find_files(model_name, scenario):
    """Find CMIP6 files for a specific model and scenario.
    
    Args:
        model_name (str): Name of the CMIP6 model
        scenario (str): Name of the scenario/experiment
        
    Returns:
        list: Paths to matching netCDF files
    """
    if scenario not in SCENARIOS:
        print(f"Warning: Unknown scenario '{scenario}'")
        return []
    
    path_component = SCENARIOS[scenario]["path_component"]
    
    search_path = os.path.join(
        CMIP6_DATA_PATH, 
        path_component, 
        "*", 
        model_name, 
        scenario, 
        "*", 
        "Omon", 
        "msftmz", 
        "*", 
        "v*", 
        f"msftmz_Omon_{model_name}_{scenario}_*.nc"
    )
    
    files = glob.glob(search_path)
    return files
''',
    "cmip6_amoc/data/processor.py": '''"""Functions for processing CMIP6 netCDF files."""

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
''',
    "cmip6_amoc/analysis/__init__.py": '"""Analysis module for CMIP6 AMOC data."""',
    "cmip6_amoc/analysis/amoc.py": '''"""Functions for calculating AMOC indices."""

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
            
            with xr.open_dataset(file) as ds:
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
''',
    "cmip6_amoc/analysis/statistics.py": '''"""Statistical analysis functions for AMOC data."""

import pandas as pd
import numpy as np

def calculate_ensemble_statistics(df):
    """Calculate ensemble statistics from a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with years as index and ensemble members as columns
        
    Returns:
        dict: Dictionary with statistics
    """
    if df is None or df.empty:
        return None
    
    stats = {
        'mean': df.mean(axis=1),
        'median': df.median(axis=1),
        'std': df.std(axis=1),
        'lower_25': df.quantile(0.25, axis=1),
        'upper_75': df.quantile(0.75, axis=1),
        'min': df.min(axis=1),
        'max': df.max(axis=1),
        'n_members': len(df.columns)
    }
    
    return stats

def calculate_trends(df, window=30):
    """Calculate trends in AMOC strength.
    
    Args:
        df (pd.DataFrame): DataFrame with years as index and ensemble members as columns
        window (int): Window size for trend calculation in years
        
    Returns:
        pd.DataFrame: DataFrame with trend values
    """
    if df is None or df.empty or len(df) < window:
        return None
    
    # Calculate ensemble mean
    mean_series = df.mean(axis=1)
    
    # Calculate rolling trends
    trends = pd.DataFrame(index=mean_series.index)
    trends['rolling_trend'] = np.nan
    
    for i in range(len(mean_series) - window + 1):
        end_idx = i + window
        years = mean_series.index[i:end_idx]
        values = mean_series.values[i:end_idx]
        
        # Simple linear regression
        slope, intercept = np.polyfit(years, values, 1)
        trends.iloc[i + window - 1, 0] = slope
    
    return trends
''',
    "cmip6_amoc/visualization/__init__.py": '"""Visualization module for CMIP6 AMOC data."""',
    "cmip6_amoc/visualization/plotting.py": '''"""Functions for visualizing AMOC data."""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ..config import SCENARIOS
from ..analysis.statistics import calculate_ensemble_statistics

def plot_multi_scenario(model_name, scenario_data):
    """Create a multi-scenario AMOC plot.
    
    Args:
        model_name (str): Name of the model
        scenario_data (dict): Dictionary mapping scenario names to DataFrames
        
    Returns:
        str: Path to saved plot file
    """
    plt.figure(figsize=(12, 8))
    
    # Track all statistics for saving
    all_stats = {}
    
    # Plot each scenario
    for scenario, df in scenario_data.items():
        if df is None or df.empty:
            print(f"No data for {scenario}, skipping")
            continue
            
        # Get scenario styling
        color = SCENARIOS.get(scenario, {}).get("color", "gray")
        linestyle = SCENARIOS.get(scenario, {}).get("linestyle", "-")
        
        # Calculate ensemble statistics
        stats = calculate_ensemble_statistics(df)
        if stats is None:
            continue
            
        all_stats[scenario] = stats
        
        # Plot ensemble mean
        plt.plot(stats['mean'].index, stats['mean'], 
                 label=f'{scenario} Mean ({stats["n_members"]} members)', 
                 color=color, linewidth=2, linestyle=linestyle)
        
        # Plot uncertainty range
        plt.fill_between(stats['lower_25'].index, stats['lower_25'], stats['upper_75'], 
                         color=color, alpha=0.2, 
                         label=f'{scenario} 25-75th percentile')
    
    # Set title and labels
    scenarios_str = ", ".join(scenario_data.keys())
    title = f"AMOC Index from {model_name} ({scenarios_str})"
    plt.title(title, fontsize=14)
    plt.ylabel("AMOC Strength (Sv)", fontsize=12)
    plt.xlabel("Year", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    # Save the plot
    output_filename = f"AMOC_index_{model_name}_multi_scenario.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Plot saved as: {output_filename}")
    
    # Save statistics to CSV
    for scenario, stats in all_stats.items():
        stats_df = pd.DataFrame({k: v for k, v in stats.items() if isinstance(v, pd.Series)})
        stats_df.to_csv(f"AMOC_index_{model_name}_{scenario}_stats.csv")
        print(f"{scenario} statistics saved as: AMOC_index_{model_name}_{scenario}_stats.csv")
    
    # Save combined means
    combined_means = pd.DataFrame({scenario: stats['mean'] for scenario, stats in all_stats.items()})
    combined_means.to_csv(f"AMOC_index_{model_name}_all_scenarios_means.csv")
    print(f"Combined means saved as: AMOC_index_{model_name}_all_scenarios_means.csv")
    
    return output_filename

def plot_multi_scenario_standardized(model_name, scenario_data):
    """Create a multi-scenario AMOC plot with standardized y-axis.
    
    Args:
        model_name (str): Name of the model
        scenario_data (dict): Dictionary mapping scenario names to DataFrames
        
    Returns:
        str: Path to saved plot file
    """
    # Standard y-axis limits for all AMOC plots (based on typical AMOC values)
    # This ensures all plots are directly comparable
    y_min = 0    # Sv
    y_max = 30   # Sv
    
    return plot_multi_scenario(model_name, scenario_data, y_limits=(y_min, y_max))

def create_multi_model_comparison(model_data, scenario):
    """Create a plot comparing AMOC across multiple models for a single scenario.
    
    Args:
        model_data (dict): Dictionary mapping model names to DataFrames
        scenario (str): Scenario name
        
    Returns:
        str: Path to saved plot file
    """
    plt.figure(figsize=(12, 8))
    
    # Plot each model
    for model_name, df in model_data.items():
        if df is None or df.empty:
            continue
            
        # Calculate ensemble statistics
        stats = calculate_ensemble_statistics(df)
        if stats is None:
            continue
        
        # Plot ensemble mean
        plt.plot(stats['mean'].index, stats['mean'], 
                 label=f'{model_name} ({stats["n_members"]} members)', 
                 linewidth=2)
    
    # Set title and labels
    title = f"AMOC Index Comparison Across Models ({scenario})"
    plt.title(title, fontsize=14)
    plt.ylabel("AMOC Strength (Sv)", fontsize=12)
    plt.xlabel("Year", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Standard y-axis limits
    plt.ylim(0, 30)
    
    plt.tight_layout()
    
    # Save the plot
    output_filename = f"AMOC_index_multi_model_comparison_{scenario}.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Plot saved as: {output_filename}")
    
    return output_filename
''',
    "scripts/__init__.py": '',
    "scripts/run_multi_scenario.py": '''#!/usr/bin/env python3
"""
Script to analyze AMOC across multiple scenarios for a single model.

Usage:
    python run_multi_scenario.py [model_name]
"""

import sys
import os
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cmip6_amoc.config import SCENARIOS
from cmip6_amoc.data.finder import find_models_with_scenarios, find_files
from cmip6_amoc.analysis.amoc import compute_amoc_index
from cmip6_amoc.visualization.plotting import plot_multi_scenario

def main():
    """Main function to run multi-scenario AMOC analysis."""
    # Get model name from command line or use default
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        # Find models with all scenarios
        available_scenarios = list(SCENARIOS.keys())
        models = find_models_with_scenarios(available_scenarios)
        
        if not models:
            print("No models found with all scenarios")
            return
        
        # Use the first available model
        model_name = models[0]
    
    print(f"\\nUsing model: {model_name}")
    
    # Process each scenario
    scenario_data = {}
    for scenario in SCENARIOS.keys():
        print(f"\\nSearching for {scenario} files for model {model_name}...")
        files = find_files(model_name, scenario)
        
        if files:
            print(f"Found {len(files)} {scenario} files for model {model_name}")
            df = compute_amoc_index(files, scenario, model_name)
            scenario_data[scenario] = df
        else:
            print(f"No {scenario} files found for model {model_name}")
    
    # Create multi-scenario plot
    if scenario_data:
        plot_multi_scenario(model_name, scenario_data)
    else:
        print("No data found for any scenario")

if __name__ == "__main__":
    main()
''',
    "scripts/run_all_models.py": '''#!/usr/bin/env python3
"""
Script to analyze AMOC across multiple scenarios for all available models.

Usage:
    python run_all_models.py
"""

import sys
import os
import pandas as pd
import concurrent.futures

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cmip6_amoc.config import SCENARIOS, RECOMMENDED_MODELS
from cmip6_amoc.data.finder import find_models_with_scenarios, find_files
from cmip6_amoc.analysis.amoc import compute_amoc_index
from cmip6_amoc.visualization.plotting import plot_multi_scenario_standardized

def process_model(model_name):
    """Process a single model and create plots."""
    print(f"\\n{'='*50}")
    print(f"Processing model: {model_name}")
    print(f"{'='*50}")
    
    # Process each scenario
    scenario_data = {}
    for scenario in SCENARIOS.keys():
        print(f"\\nSearching for {scenario} files for model {model_name}...")
        files = find_files(model_name, scenario)
        
        if files:
            print(f"Found {len(files)} {scenario} files for model {model_name}")
            df = compute_amoc_index(files, scenario, model_name)
            scenario_data[scenario] = df
        else:
            print(f"No {scenario} files found for model {model_name}")
    
    # Create multi-scenario plot
    if scenario_data:
        plot_multi_scenario_standardized(model_name, scenario_data)
        return True
    else:
        print(f"No data found for any scenario for model {model_name}")
        return False

def main():
    """Main function to run AMOC analysis for all models."""
    # Find models with at least historical and ssp585 data
    print("Finding models with both historical and SSP585 data...")
    available_models = find_models_with_scenarios(["historical", "ssp585"])
    
    if not available_models:
        print("No models found with required scenarios")
        return
    
    print(f"\\nFound {len(available_models)} models with required data:")
    for i, model in enumerate(available_models, 1):
        print(f"{i}. {model}")
    
    # Process all models
    results = {}
    
    # Option 1: Sequential processing
    for model in available_models:
        success = process_model(model)
        results[model] = success
    
    # Option 2: Parallel processing (uncomment to use)
    # with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
    #     future_to_model = {executor.submit(process_model, model): model for model in available_models}
    #     for future in concurrent.futures.as_completed(future_to_model):
    #         model = future_to_model[future]
    #         try:
    #             success = future.result()
    #             results[model] = success
    #         except Exception as e:
    #             print(f"Error processing {model}: {e}")
    #             results[model] = False
    
    # Print summary
    print("\\n" + "="*50)
    print("Processing Summary:")
    print("="*50)
    successful = [model for model, success in results.items() if success]
    failed = [model for model, success in results.items() if not success]
    
    print(f"Successfully processed {len(successful)} models:")
    for model in successful:
        print(f"  - {model}")
    
    if failed:
        print(f"\\nFailed to process {len(failed)} models:")
        for model in failed:
            print(f"  - {model}")
    
    print("\\nAll plots have been created with standardized y-axis scales.")

if __name__ == "__main__":
    main()
''',
    "run_all_models.sh": '''#!/bin/bash
#SBATCH --job-name=amoc_all_models
#SBATCH --output=amoc_all_models_%j.out
#SBATCH --error=amoc_all_models_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --account=kd1418
#SBATCH --partition=compute
#SBATCH --time=12:00:00
#SBATCH --mem=64G

# Load necessary modules
module load clint codes

# Run the Python script for all models
python scripts/run_all_models.py
'''
}

# Create the package structure
def create_package():
    """Create the package structure and files."""
    for file_path, content in package_structure.items():
        # Create directory if it doesn't exist
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        
        # Write file content
        with open(file_path, 'w') as f:
            f.write(content)
        
        print(f"Created file: {file_path}")

if __name__ == "__main__":
    create_package()
    print("\nPackage structure created successfully!")
    print("To run the multi-scenario analysis, use:")
    print("sbatch run_amoc.sh [model_name]")