#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import logging
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import cftime
import freva
import matplotlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import Freva modules with better error handling
try:
    from freva import databrowser
    logger.info("Successfully imported Freva databrowser")
except ImportError as e:
    logger.error(f"Error importing Freva databrowser: {str(e)}")
    logger.error("Please make sure Freva is installed and properly configured.")
    sys.exit(1)
except Exception as e:
    logger.error(f"Unexpected error importing Freva: {str(e)}")
    logger.error(traceback.format_exc())
    sys.exit(1)

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cmip6_amoc.config import SCENARIOS, RECOMMENDED_MODELS
from cmip6_amoc.data.freva_finder import find_models_with_scenarios_freva, find_files_freva
from cmip6_amoc.analysis.amoc import compute_amoc_index
from cmip6_amoc.visualization.plotting import plot_multi_scenario_standardized, plot_amoc_percentage_changes

# Comprehensive list of CMIP6 models that might have AMOC data
ALL_CMIP6_MODELS = [
    # Models from RECOMMENDED_MODELS
    "CanESM5", "MIROC6", "MPI-ESM1-2-LR", "NorESM2-LM", 
    "UKESM1-0-LL", "CESM2", "GFDL-ESM4", "EC-Earth3",
    
    # Additional models that were successfully processed
    "CESM2-WACCM", "CanESM5-CanOE", "FGOALS-f3-L", "FGOALS-g3", 
    "GISS-E2-1-G", "INM-CM4-8", "INM-CM5-0", "MPI-ESM1-2-HR", 
    "MRI-ESM2-0", "NorESM2-MM",
    
    # More CMIP6 models to try
    "ACCESS-CM2", "ACCESS-ESM1-5", "AWI-CM-1-1-MR", "BCC-CSM2-MR",
    "BCC-ESM1", "CAMS-CSM1-0", "CAS-ESM2-0", "CESM2-FV2", 
    "CESM2-WACCM-FV2", "CIESM", "CMCC-CM2-HR4", "CMCC-CM2-SR5",
    "CMCC-ESM2", "CNRM-CM6-1", "CNRM-CM6-1-HR", "CNRM-ESM2-1",
    "E3SM-1-0", "E3SM-1-1", "E3SM-1-1-ECA", "EC-Earth3-AerChem",
    "EC-Earth3-CC", "EC-Earth3-LR", "EC-Earth3-Veg", "EC-Earth3-Veg-LR",
    "FIO-ESM-2-0", "GFDL-CM4", "GFDL-CM4C192", "GFDL-ESM4",
    "GISS-E2-1-G-CC", "GISS-E2-1-H", "GISS-E2-2-G", "GISS-E2-2-H",
    "HadGEM3-GC31-LL", "HadGEM3-GC31-MM", "IITM-ESM", "IPSL-CM5A2-INCA",
    "IPSL-CM6A-LR", "IPSL-CM6A-LR-INCA", "KACE-1-0-G", "KIOST-ESM",
    "MCM-UA-1-0", "MIROC-ES2H", "MIROC-ES2L", "MPI-ESM-1-2-HAM",
    "MPI-ESM1-2-HR", "NESM3", "NorCPM1", "NorESM1-F", 
    "SAM0-UNICON", "TaiESM1", "UKESM1-0-LL", "UKESM1-1-LL"
]

# List of CMIP5 models that might have AMOC data
ALL_CMIP5_MODELS = [
    "ACCESS1-0", "ACCESS1-3", "BCC-CSM1-1", "BCC-CSM1-1-m", "BNU-ESM",
    "CanESM2", "CCSM4", "CESM1-BGC", "CESM1-CAM5", "CESM1-WACCM",
    "CMCC-CESM", "CMCC-CM", "CMCC-CMS", "CNRM-CM5", "CSIRO-Mk3-6-0",
    "EC-EARTH", "FGOALS-g2", "FGOALS-s2", "FIO-ESM", "GFDL-CM3",
    "GFDL-ESM2G", "GFDL-ESM2M", "GISS-E2-H", "GISS-E2-H-CC", "GISS-E2-R",
    "GISS-E2-R-CC", "HadGEM2-AO", "HadGEM2-CC", "HadGEM2-ES", "INM-CM4",
    "IPSL-CM5A-LR", "IPSL-CM5A-MR", "IPSL-CM5B-LR", "MIROC-ESM",
    "MIROC-ESM-CHEM", "MIROC5", "MPI-ESM-LR", "MPI-ESM-MR", "MPI-ESM-P",
    "MRI-CGCM3", "MRI-ESM1", "NorESM1-M", "NorESM1-ME"
]

# All CMIP6 SSP scenarios to check
ALL_CMIP6_SCENARIOS = ["ssp119", "ssp126", "ssp245", "ssp370", "ssp434", "ssp460", "ssp534-over", "ssp585"]

# All CMIP5 RCP scenarios to check
ALL_CMIP5_SCENARIOS = ["rcp26", "rcp45", "rcp60", "rcp85"]

# Define models and experiments
#CMIP6_MODELS = [
#    "MPI-ESM1-2-LR"  # Only include MPI-ESM1-2-LR for testing
#]
CMIP6_MODELS =  ALL_CMIP6_MODELS

CMIP6_EXPERIMENTS = [
    "historical",
    "ssp126",
    "ssp245",
    "ssp370",
    "ssp585"
]

def find_available_scenarios(model_name, project="CMIP6"):
    """Find all available scenarios for a model.
    
    Args:
        model_name (str): Name of the model
        project (str): Project name (CMIP5 or CMIP6)
        
    Returns:
        dict: Dictionary mapping scenario names to file lists
    """
    scenario_files = {}
    
    # Always check historical
    historical_files = find_files_freva(model_name, "historical", project)
    if historical_files:
        scenario_files["historical"] = historical_files
    
    # Check all future scenarios based on project
    if project == "CMIP5":
        scenarios_to_check = ALL_CMIP5_SCENARIOS
    else:  # CMIP6
        scenarios_to_check = ALL_CMIP6_SCENARIOS
    
    for scenario in scenarios_to_check:
        files = find_files_freva(model_name, scenario, project)
        if files:
            scenario_files[scenario] = files
    
    return scenario_files

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process CMIP5 and CMIP6 model data for AMOC analysis')
    parser.add_argument('--cmip5', action='store_true', help='Process CMIP5 models')
    parser.add_argument('--cmip6', action='store_true', help='Process CMIP6 models')
    parser.add_argument('--all', action='store_true', help='Process all models')
    parser.add_argument('--end-year', type=int, default=2100, help='End year for analysis')
    parser.add_argument('--output-dir', type=str, default='output', help='Directory for output files')
    parser.add_argument('--model', type=str, help='Analyze only this specific model')
    return parser.parse_args()

def run_freva_search(model_name, experiment, project=None):
    """
    Run a Freva search using bash commands.
    
    Parameters:
    -----------
    model_name : str
        Name of the climate model
    experiment : str
        Name of the experiment (e.g., 'historical', 'ssp585')
    project : str, optional
        Project name (e.g., 'CMIP5', 'CMIP6')
        
    Returns:
    --------
    list
        List of file paths matching the search criteria
    """
    try:
        # Construct the freva search command with correct syntax
        cmd = ["freva", "databrowser", f"variable=msftmz", f"model={model_name}", 
               f"experiment={experiment}", "time_frequency=mon"]
        
        if project:
            cmd.append(f"project={project}")
            
        logger.debug(f"Running command: {' '.join(cmd)}")
        
        # Run the command and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        if result.returncode != 0:
            logger.error(f"Error running freva search: {result.stderr}")
            return []
            
        # Parse the output to get file paths
        file_paths = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if line and line.endswith('.nc'):
                file_paths.append(line)
                
        logger.debug(f"Found {len(file_paths)} files")
        return file_paths
        
    except Exception as e:
        logger.error(f"Error running freva search: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def get_model_data(model_name, experiment, end_year):
    """
    Retrieve AMOC data for a specific model and experiment.
    
    Parameters:
    -----------
    model_name : str
        Name of the climate model
    experiment : str
        Name of the experiment (e.g., 'historical', 'ssp585')
    end_year : int
        End year for the analysis
        
    Returns:
    --------
    dict or None
        Dictionary of DataFrames for each ensemble member or None if data not available
    """
    try:
        # Construct the freva databrowser command
        cmd = f"freva databrowser project=CMIP6 model={model_name} experiment={experiment} variable=msftmz"
        logger.info(f"Running command: {cmd}")
        
        # Execute the command
        import subprocess
        try:
            result = subprocess.check_output(cmd, shell=True, universal_newlines=True)
            file_paths = [line.strip() for line in result.split('\n') if line.strip()]
        except subprocess.CalledProcessError as e:
            logger.warning(f"Error running freva databrowser: {str(e)}")
            return None
        
        if not file_paths:
            logger.warning(f"No data files found for {model_name} - {experiment}")
            return None
        
        # Group files by ensemble member
        import re
        member_pattern = r'r\d+i\d+p\d+f\d+'
        member_files = {}
        
        for file_path in file_paths:
            match = re.search(member_pattern, file_path)
            if match:
                member_id = match.group(0)
                if member_id not in member_files:
                    member_files[member_id] = []
                member_files[member_id].append(file_path)
        
        if not member_files:
            logger.warning(f"Could not identify ensemble members for {model_name} - {experiment}")
            # Fall back to using all files
            member_files = {"unknown": file_paths}
        
        # Log the number of members found
        logger.info(f"Found {len(member_files)} ensemble members for {model_name} - {experiment}: {', '.join(member_files.keys())}")
        
        # Process each member
        all_members_data = {}
        
        for member_id, files in member_files.items():
            logger.info(f"Processing member {member_id} with {len(files)} files")
            
            # Sort files to ensure chronological order
            files.sort()
            
            # Open and merge files for this member
            time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
            ds_list = []
            
            for file_path in files:
                try:
                    ds = xr.open_dataset(file_path, decode_times=time_coder)
                    ds_list.append(ds)
                except Exception as e:
                    logger.warning(f"Error opening file {file_path}: {str(e)}")
            
            if not ds_list:
                logger.warning(f"No valid files for member {member_id}")
                continue
                
            # Merge datasets along time dimension
            try:
                if len(ds_list) > 1:
                    ds = xr.concat(ds_list, dim='time')
                    logger.info(f"Successfully merged {len(ds_list)} files for member {member_id}")
                else:
                    ds = ds_list[0]
            except Exception as e:
                logger.warning(f"Error merging datasets for member {member_id}: {str(e)}")
                # Try using just the first file
                ds = ds_list[0]
                logger.info(f"Falling back to using just the first file for member {member_id}")
            
            # Process this member's data
            try:
                # Check if msftmz variable exists
                if 'msftmz' in ds:
                    # Print variable attributes to debug
                    logger.info(f"msftmz attributes: {ds.msftmz.attrs}")
                    if 'units' in ds.msftmz.attrs:
                        logger.info(f"msftmz units: {ds.msftmz.attrs['units']}")
                    
                    # Find the latitude closest to 26.5°N
                    target_lat = 26.5
                    lat_idx = abs(ds.lat.values - target_lat).argmin()
                    actual_lat = ds.lat.values[lat_idx]
                    logger.info(f"Using latitude index {lat_idx} ({actual_lat}°N)")
                    
                    # Find the maximum of the streamfunction at each time step
                    # This is the correct way to extract the AMOC strength
                    if 'lev' in ds.dims:
                        # Find the Atlantic basin index if it exists
                        basin_idx = None
                        if 'basin' in ds.dims:
                            # Try index 0 first for Atlantic
                            basin_idx = 0
                            logger.info("Using basin index 0 (assuming it's Atlantic)")
                        
                        # Extract AMOC data
                        if basin_idx is not None:
                            # Get the streamfunction at the specified latitude and basin
                            amoc_profile = ds.msftmz.isel(lat=lat_idx, basin=basin_idx)
                            
                            # Find the maximum value at each time step (this is the AMOC strength)
                            amoc = amoc_profile.max(dim='lev')
                            logger.info(f"Extracted AMOC as maximum over depth at lat={actual_lat}°N, basin={basin_idx}")
                            
                            # Check if values are reasonable (around 15-25 Sv)
                            mean_val = float(amoc.mean())
                            logger.info(f"Mean AMOC value before unit conversion: {mean_val}")
                            
                            # If values seem too small or too large, try other basin indices
                            if abs(mean_val) < 5 or abs(mean_val) > 100:
                                logger.warning(f"AMOC values with basin index 0 seem unreasonable: {mean_val}. Trying other indices...")
                                
                                # Try other basin indices
                                for test_idx in [1, 2]:
                                    if test_idx < len(ds.basin.values):
                                        try:
                                            test_profile = ds.msftmz.isel(lat=lat_idx, basin=test_idx)
                                            test_amoc = test_profile.max(dim='lev')
                                            test_mean = float(test_amoc.mean())
                                            logger.info(f"Basin index {test_idx} mean before unit conversion: {test_mean}")
                                            
                                            # If this index gives more reasonable values, use it
                                            if 5 <= abs(test_mean) <= 50:
                                                amoc = test_amoc
                                                basin_idx = test_idx
                                                logger.info(f"Switched to basin index {test_idx} with mean {test_mean}")
                                                break
                                        except:
                                            continue
                        else:
                            # Try to find a dimension that might represent the Atlantic
                            basin_found = False
                            
                            # Check for other possible basin dimensions
                            for dim in ds.dims:
                                if dim.lower() in ['basin', 'basin_index', 'ocean_basin', 'basin_id', 'region']:
                                    # Try index 0 first, then others
                                    for idx in [0, 1, 2]:
                                        if idx < ds.dims[dim]:
                                            try:
                                                amoc_profile = ds.msftmz.isel(lat=lat_idx, **{dim: idx})
                                                amoc = amoc_profile.max(dim='lev')
                                                mean_val = float(amoc.mean())
                                                
                                                # Check if values are reasonable
                                                if 5 <= abs(mean_val) <= 50:
                                                    basin_found = True
                                                    logger.info(f"Using {dim} index {idx}, max over depth, mean before unit conversion={mean_val}")
                                                    break
                                                else:
                                                    logger.info(f"{dim} index {idx} has mean before unit conversion={mean_val}, trying next")
                                            except:
                                                continue
                                    
                                    if basin_found:
                                        break
                            
                            if not basin_found:
                                # If no basin dimension found, try using the whole array
                                try:
                                    amoc_profile = ds.msftmz.isel(lat=lat_idx)
                                    amoc = amoc_profile.max(dim='lev')
                                    logger.info("Using maximum over depth without basin selection")
                                except:
                                    logger.warning(f"Could not extract AMOC for {model_name} - {experiment} member {member_id}")
                                    continue
                    else:
                        # No depth dimension, try to use just latitude
                        amoc = ds.msftmz.isel(lat=lat_idx)
                        logger.warning("No depth dimension found, using values at specified latitude only")
                    
                    # Check units and convert if needed
                    if 'units' in ds.msftmz.attrs:
                        units = ds.msftmz.attrs['units'].lower()
                        logger.info(f"Original units: {units}")
                        
                        # Convert to Sv if needed
                        if 'kg/s' in units or 'kg s-1' in units:
                            # Convert from kg/s to Sv (1 Sv = 10^6 m^3/s)
                            # Assuming density of seawater is about 1025 kg/m^3
                            amoc = amoc / (1.025e9)  # 1.025e9 = 1025 kg/m^3 * 10^6 m^3/s
                            logger.info("Converted from kg/s to Sv by dividing by 1.025e9")
                        elif units == 'm3/s' or units == 'm3 s-1':
                            # Convert from m^3/s to Sv (1 Sv = 10^6 m^3/s)
                            amoc = amoc / 1e6
                            logger.info("Converted from m^3/s to Sv by dividing by 1e6")
                    else:
                        # If no units are specified, assume kg/s and convert to Sv
                        logger.warning("No units specified, assuming kg/s and converting to Sv")
                        amoc = amoc / (1.025e9)
                    
                    # Sample a few values to check if they're reasonable
                    sample_values = amoc.values
                    if len(sample_values) > 0:
                        sample_mean = np.mean(sample_values)
                        logger.info(f"Sample AMOC values after unit conversion: mean={sample_mean:.2f} Sv")
                        
                        # Just log a warning if values are still unreasonable, but don't modify them
                        if abs(sample_mean) > 100:
                            logger.warning(f"AMOC values seem unusually large after conversion: {sample_mean:.2f} Sv. Please check the data source.")
                        elif abs(sample_mean) < 5:
                            logger.warning(f"AMOC values seem unusually small after conversion: {sample_mean:.2f} Sv. Please check the data source.")
                    
                    # Extract year from time coordinate
                    if hasattr(amoc, 'time'):
                        # Add a year coordinate for groupby operations
                        years = np.array([t.year for t in amoc.time.values])
                        amoc = amoc.assign_coords(year=('time', years))
                        
                        # Calculate annual means directly in xarray
                        annual_amoc = amoc.groupby('year').mean()
                        logger.info(f"Calculated annual means from {len(amoc)} monthly values to {len(annual_amoc)} yearly values")
                        
                        # Convert to DataFrame for easier handling
                        df = annual_amoc.to_dataframe().reset_index()
                        
                        # Ensure we have a 'year' column
                        if 'year' not in df.columns:
                            logger.warning("Year column missing after groupby operation")
                            continue
                    else:
                        # Convert to DataFrame first
                        df = amoc.to_dataframe().reset_index()
                        
                        # Extract year from time
                        if 'time' in df.columns:
                            if isinstance(df['time'].iloc[0], cftime._cftime.DatetimeNoLeap) or \
                               isinstance(df['time'].iloc[0], cftime._cftime.Datetime360Day) or \
                               isinstance(df['time'].iloc[0], cftime._cftime.DatetimeGregorian) or \
                               isinstance(df['time'].iloc[0], cftime._cftime.DatetimeJulian) or \
                               isinstance(df['time'].iloc[0], cftime._cftime.DatetimeAllLeap) or \
                               isinstance(df['time'].iloc[0], cftime._cftime.DatetimeProlepticGregorian):
                                df['year'] = df['time'].apply(lambda x: x.year)
                            else:
                                # Try to convert to pandas datetime
                                try:
                                    df['time'] = pd.to_datetime(df['time'])
                                    df['year'] = df['time'].dt.year
                                except:
                                    logger.error(f"Could not convert time to datetime for {model_name} - {experiment} member {member_id}")
                                    continue
                        else:
                            logger.warning(f"No time dimension found for {model_name} - {experiment} member {member_id}")
                            continue
                        
                        # Calculate annual means
                        annual_means = df.groupby('year')['msftmz'].mean().reset_index()
                        df = annual_means
                        logger.info(f"Calculated annual means in pandas from {len(amoc)} values to {len(df)} yearly values")
                    
                    # Filter by end year
                    df = df[df['year'] <= end_year]
                    
                    if df.empty:
                        logger.warning(f"No data available for {model_name} - {experiment} member {member_id} after filtering")
                        continue
                    
                    # We'll apply the running mean in the main function when calculating ensemble statistics
                    # This ensures we have the raw annual data for each member
                    
                    # Final check for reasonable values - just log warnings, don't modify data
                    mean_val = df['msftmz'].mean()
                    if abs(mean_val) < 5 or abs(mean_val) > 50:
                        logger.warning(f"Final AMOC values seem unusual: mean={mean_val:.2f} Sv. Please verify the results.")
                    
                    # Check if AMOC is decreasing in future scenarios as expected
                    if experiment.startswith('ssp') and len(df) > 10:
                        start_val = df['msftmz'].iloc[:10].mean()
                        end_val = df['msftmz'].iloc[-10:].mean()
                        change = end_val - start_val
                        logger.info(f"{model_name} {experiment} {member_id}: AMOC change = {change:.2f} Sv (start: {start_val:.2f}, end: {end_val:.2f})")
                        
                        if change > 0 and abs(change) > 2:
                            logger.warning(f"WARNING: AMOC is increasing in {experiment} for {model_name} {member_id}. This is unexpected but could be model-specific behavior.")
                    
                    # Add this member's data to the dictionary
                    all_members_data[member_id] = df
                    logger.info(f"Successfully processed member {member_id} with {len(df)} years of data")
                    
                else:
                    logger.warning(f"No msftmz variable found in dataset for {model_name} - {experiment} member {member_id}")
            except Exception as e:
                logger.error(f"Error processing {model_name} - {experiment} member {member_id}: {str(e)}")
                logger.error(traceback.format_exc())
        
        if all_members_data:
            logger.info(f"Successfully processed {len(all_members_data)} members for {model_name} - {experiment}")
            return all_members_data
        else:
            logger.warning(f"No valid data found for any member of {model_name} - {experiment}")
            return None
            
    except Exception as e:
        logger.error(f"Error getting data for {model_name} - {experiment}: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def process_model(model_name, experiment, output_dir, end_year):
    """
    Process data for a specific model and experiment.
    
    Parameters:
    -----------
    model_name : str
        Name of the climate model
    experiment : str
        Name of the experiment (e.g., 'historical', 'ssp585')
    output_dir : str
        Directory for output files
    end_year : int
        End year for the analysis
        
    Returns:
    --------
    tuple
        (success, DataFrame) where success is a boolean indicating whether processing was successful,
        and DataFrame contains the processed data (or None if processing failed)
    """
    try:
        logger.info(f"Processing {model_name} - {experiment}")
        
        # Get data for the model and experiment
        df = get_model_data(model_name, experiment, end_year)
        
        if df is None or df.empty:
            logger.warning(f"No data available for {model_name} - {experiment}")
            return False, None
            
        # Save processed data
        output_file = os.path.join(output_dir, f"{model_name}_{experiment}.csv")
        df.to_csv(output_file, index=False)
        logger.info(f"Successfully processed {model_name} - {experiment}")
        return True, df
            
    except Exception as e:
        logger.error(f"Error processing {model_name} - {experiment}: {str(e)}")
        logger.error(traceback.format_exc())
        return False, None

def plot_amoc_timeseries(all_model_data, output_dir, project):
    """
    Plot AMOC timeseries for all models.
    
    Parameters:
    -----------
    all_model_data : dict
        Dictionary containing model data
    output_dir : str
        Directory for output files
    project : str
        Project name (e.g., 'CMIP5', 'CMIP6')
    """
    try:
        # Create plots directory
        plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Define colors for scenarios
        scenario_colors = {
            'historical': 'black',
            'rcp26': 'blue',
            'rcp45': 'orange',
            'rcp85': 'red',
            'ssp119': 'lightblue',
            'ssp126': 'blue',
            'ssp245': 'orange',
            'ssp370': 'darkred',
            'ssp585': 'red'
        }
        
        # Group data by model
        model_data = {}
        for model_scenario, data in all_model_data.items():
            model, scenario = model_scenario.split(' - ')
            if model not in model_data:
                model_data[model] = {}
            model_data[model][scenario] = data
        
        # Plot multi-model mean
        plt.figure(figsize=(12, 8))
        
        # Group data by scenario
        scenario_data = {}
        for model_scenario, data in all_model_data.items():
            _, scenario = model_scenario.split(' - ')
            if scenario not in scenario_data:
                scenario_data[scenario] = []
            scenario_data[scenario].append(data)
        
        # Calculate and plot multi-model mean for each scenario
        for scenario, data_list in scenario_data.items():
            if not data_list:
                continue
                
            # Find common years across all models
            common_years = set(data_list[0]['year'])
            for data in data_list[1:]:
                common_years &= set(data['year'])
            
            if not common_years:
                logger.warning(f"No common years found for scenario {scenario}")
                continue
                
            common_years = sorted(common_years)
            
            # Calculate multi-model mean
            multi_model_mean = pd.DataFrame({'year': common_years})
            multi_model_mean['msftmz'] = 0.0
            
            for data in data_list:
                filtered_data = data[data['year'].isin(common_years)]
                multi_model_mean['msftmz'] += filtered_data['msftmz'].values
            
            multi_model_mean['msftmz'] /= len(data_list)
            
            # Plot multi-model mean
            plt.plot(multi_model_mean['year'], multi_model_mean['msftmz'], 
                     label=f"{scenario} (n={len(data_list)})", 
                     color=scenario_colors.get(scenario, 'gray'),
                     linewidth=2)
        
        plt.title(f'{project} Multi-Model Mean AMOC Strength at 26.5°N, ~1000m')
        plt.xlabel('Year')
        plt.ylabel('AMOC Strength (Sv)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        output_file = os.path.join(plots_dir, f'{project}_multi_model_mean_amoc_timeseries.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved multi-model mean plot to {output_file}")
        plt.close()
        
        # Plot individual scenarios
        for scenario in scenario_data.keys():
            if scenario not in scenario_data or not scenario_data[scenario]:
                continue
                
            plt.figure(figsize=(12, 8))
            
            # Plot each model
            for data in scenario_data[scenario]:
                model = next(key.split(' - ')[0] for key, val in all_model_data.items() 
                             if val is data)
                plt.plot(data['year'], data['msftmz'], alpha=0.5, linewidth=1)
            
            # Plot multi-model mean
            common_years = set(scenario_data[scenario][0]['year'])
            for data in scenario_data[scenario][1:]:
                common_years &= set(data['year'])
            
            if common_years:
                common_years = sorted(common_years)
                
                multi_model_mean = pd.DataFrame({'year': common_years})
                multi_model_mean['msftmz'] = 0.0
                
                for data in scenario_data[scenario]:
                    filtered_data = data[data['year'].isin(common_years)]
                    multi_model_mean['msftmz'] += filtered_data['msftmz'].values
                
                multi_model_mean['msftmz'] /= len(scenario_data[scenario])
                
                plt.plot(multi_model_mean['year'], multi_model_mean['msftmz'], 
                         label='Multi-Model Mean', 
                         color='black',
                         linewidth=3)
            
            plt.title(f'{project} {scenario} AMOC Strength at 26.5°N, ~1000m (n={len(scenario_data[scenario])})')
            plt.xlabel('Year')
            plt.ylabel('AMOC Strength (Sv)')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            output_file = os.path.join(plots_dir, f'{project}_{scenario}_amoc_timeseries.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved {scenario} plot to {output_file}")
            plt.close()
        
        # Find global min and max for consistent y-axis
        y_min = float('inf')
        y_max = float('-inf')
        
        for model, scenarios in model_data.items():
            for scenario, data in scenarios.items():
                y_min = min(y_min, data['msftmz'].min())
                y_max = max(y_max, data['msftmz'].max())
        
        # Add some padding
        y_range = y_max - y_min
        y_min -= 0.1 * y_range
        y_max += 0.1 * y_range
        
        # Plot individual models
        for model, scenarios in model_data.items():
            plt.figure(figsize=(12, 8))
            
            for scenario, data in scenarios.items():
                # Count ensemble members for this model and scenario
                ensemble_count = sum(1 for key in all_model_data.keys() 
                                    if key.startswith(f"{model} - {scenario}"))
                
                plt.plot(data['year'], data['msftmz'], 
                         label=f"{scenario} (n={ensemble_count})", 
                         color=scenario_colors.get(scenario, 'gray'),
                         linewidth=2)
            
            plt.title(f'{project} {model} AMOC Strength at 26.5°N, ~1000m')
            plt.xlabel('Year')
            plt.ylabel('AMOC Strength (Sv)')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.ylim(y_min, y_max)
            
            output_file = os.path.join(plots_dir, f'{project}_{model}_amoc_timeseries.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {output_file}")
            plt.close()
            
    except Exception as e:
        logger.error(f"Error plotting AMOC timeseries: {str(e)}")
        logger.error(traceback.format_exc())

def plot_amoc_percentage_changes(all_model_data, output_dir, project):
    """
    Plot AMOC percentage changes relative to 1850-1900 for all models.
    
    Parameters:
    -----------
    all_model_data : dict
        Dictionary containing model data
    output_dir : str
        Directory for output files
    project : str
        Project name (e.g., 'CMIP5', 'CMIP6')
    """
    try:
        # Create plots directory if it doesn't exist
        plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Calculate baseline (1850-1900 mean) for each model
        baselines = {}
        for model, model_data in all_model_data.items():
            if 'historical' in model_data:
                df = model_data['historical']
                baseline_data = df[(df['year'] >= 1850) & (df['year'] <= 1900)]
                if not baseline_data.empty:
                    baselines[model] = baseline_data['msftmz'].mean()
        
        # Define future scenarios
        future_scenarios = [s for s in ['rcp26', 'rcp45', 'rcp60', 'rcp85', 'ssp126', 'ssp245', 'ssp370', 'ssp585'] 
                           if any(s in model_data for model_data in all_model_data.values())]
        
        # Define colors for scenarios
        scenario_colors = {
            'rcp26': 'blue',
            'rcp45': 'green',
            'rcp60': 'orange',
            'rcp85': 'red',
            'ssp126': 'blue',
            'ssp245': 'green',
            'ssp370': 'orange',
            'ssp585': 'red'
        }
        
        # Find common y-axis limits for all plots
        all_percentages = []
        for model, model_data in all_model_data.items():
            if model in baselines:
                baseline = baselines[model]
                for scenario in future_scenarios:
                    if scenario in model_data:
                        df = model_data[scenario]
                        percentage_change = ((df['msftmz'] - baseline) / baseline) * 100
                        all_percentages.extend(percentage_change.values)
        
        if all_percentages:
            y_min = np.floor(min(all_percentages) - 5)
            y_max = np.ceil(max(all_percentages) + 5)
        else:
            y_min, y_max = -30, 10  # Default range if no data
        
        # Create a combined plot for all scenarios
        plt.figure(figsize=(12, 8))
        
        # Group data by scenario
        for scenario in future_scenarios:
            scenario_data = []
            years_set = set()
            
            for model, model_data in all_model_data.items():
                if model in baselines and scenario in model_data:
                    baseline = baselines[model]
                    df = model_data[scenario]
                    percentage_change = ((df['msftmz'] - baseline) / baseline) * 100
                    scenario_data.append(pd.DataFrame({'year': df['year'], 'percentage': percentage_change}))
                    years_set.update(df['year'].values)
            
            if scenario_data:
                # Get common years
                years = sorted(years_set)
                
                # Calculate mean and std for each year
                mean_values = []
                lower_values = []
                upper_values = []
                
                for year in years:
                    year_values = []
                    for df in scenario_data:
                        if year in df['year'].values:
                            value = df.loc[df['year'] == year, 'percentage'].values[0]
                            year_values.append(value)
                    
                    if year_values:
                        mean_values.append(np.mean(year_values))
                        std = np.std(year_values)
                        lower_values.append(np.mean(year_values) - std)
                        upper_values.append(np.mean(year_values) + std)
                
                # Plot mean and uncertainty
                plt.plot(years, mean_values, color=scenario_colors.get(scenario, 'gray'), 
                         linewidth=2, label=scenario)
                plt.fill_between(years, lower_values, upper_values, 
                                 color=scenario_colors.get(scenario, 'gray'), alpha=0.2)
        
        # Add legend, title, and labels
        plt.legend(loc='lower left')
        plt.title(f'AMOC Strength Percentage Change Relative to 1850-1900 - {project} - Multi-Model Mean')
        plt.xlabel('Year')
        plt.ylabel('Percentage Change (%)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.ylim(y_min, y_max)
        
        # Save the figure to both directories for compatibility
        output_file = os.path.join(output_dir, f'amoc_percentage_changes_{project.lower()}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {output_file}")
        
        plots_output_file = os.path.join(plots_dir, f'{project}_amoc_percentage_change_all.png')
        plt.savefig(plots_output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {plots_output_file}")
        
        plt.close()
        
        # Create individual plots for each scenario
        for scenario in future_scenarios:
            plt.figure(figsize=(12, 8))
            
            for model, model_data in all_model_data.items():
                if model in baselines and scenario in model_data:
                    baseline = baselines[model]
                    df = model_data[scenario]
                    percentage_change = ((df['msftmz'] - baseline) / baseline) * 100
                    plt.plot(df['year'], percentage_change, alpha=0.7, linewidth=1, label=model)
            
            plt.legend(loc='lower left')
            plt.title(f'AMOC Strength Percentage Change Relative to 1850-1900 - {project} - {scenario}')
            plt.xlabel('Year')
            plt.ylabel('Percentage Change (%)')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.ylim(y_min, y_max)
            
            output_file = os.path.join(plots_dir, f'{project}_amoc_percentage_change_{scenario}.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {output_file}")
            plt.close()
        
        # Create individual plots for each model
        for model, model_data in all_model_data.items():
            if model in baselines:
                plt.figure(figsize=(12, 8))
                baseline = baselines[model]
                
                for scenario in future_scenarios:
                    if scenario in model_data:
                        df = model_data[scenario]
                        percentage_change = ((df['msftmz'] - baseline) / baseline) * 100
                        plt.plot(df['year'], percentage_change, color=scenario_colors.get(scenario, 'gray'), 
                                linewidth=2, label=scenario)
                
                plt.legend(loc='lower left')
                plt.title(f'AMOC Strength Percentage Change Relative to 1850-1900 - {model}')
                plt.xlabel('Year')
                plt.ylabel('Percentage Change (%)')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                plt.ylim(y_min, y_max)
                
                output_file = os.path.join(plots_dir, f'{project}_{model}_amoc_percentage_change.png')
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                logger.info(f"Saved plot to {output_file}")
                plt.close()
            
    except Exception as e:
        logger.error(f"Error plotting AMOC percentage changes: {str(e)}")
        logger.error(traceback.format_exc())

def process_cmip6_data(output_dir, end_year):
    """
    Process CMIP6 data.
    
    Parameters:
    -----------
    output_dir : str
        Directory for output files
    end_year : int
        End year for analysis
    """
    logger.info(f"Processing CMIP6 data with end year {end_year}")
    
    # Define CMIP6 models and experiments
    models = CMIP6_MODELS
    experiments = CMIP6_EXPERIMENTS
    
    # Process data for each model and experiment
    all_model_data = {}
    
    for model_name in models:
        for experiment in experiments:
            logger.info(f"Processing {model_name} - {experiment}")
            
            # Get model data
            model_data = get_model_data(model_name, experiment, end_year)
            
            if model_data is not None:
                all_model_data[f"{model_name} - {experiment}"] = model_data
            else:
                logger.warning(f"No data available for {model_name} - {experiment}")
    
    # Plot results
    if all_model_data:
        logger.info(f"Successfully processed {len(all_model_data)} models:")
        for model_scenario in all_model_data.keys():
            logger.info(f"  - {model_scenario}")
        
        plot_amoc_timeseries(all_model_data, output_dir, 'CMIP6')
        logger.info("All plots have been created with standardized y-axis scales.")
    else:
        logger.warning("No data available for any model/experiment combination.")

def get_experiment_color(experiment):
    """
    Get a consistent color for each experiment.
    
    Parameters:
    -----------
    experiment : str
        Name of the experiment
        
    Returns:
    --------
    str
        Color code
    """
    colors = {
        'historical': 'black',
        'ssp119': 'blue',
        'ssp126': 'green',
        'ssp245': 'orange',
        'ssp370': 'red',
        'ssp434': 'purple',
        'ssp460': 'brown',
        'ssp534-over': 'pink',
        'ssp585': 'darkred'
    }
    
    return colors.get(experiment, 'gray')

def get_model_color(model_name):
    """
    Get a consistent color for a given model.
    
    Parameters:
    -----------
    model_name : str
        Name of the model
        
    Returns:
    --------
    tuple
        RGB color tuple
    """
    # Use a hash function to generate a color based on the model name
    import hashlib
    
    # Get a hash of the model name
    hash_obj = hashlib.md5(model_name.encode())
    hash_hex = hash_obj.hexdigest()
    
    # Convert the first 6 characters of the hash to RGB values
    r = int(hash_hex[0:2], 16) / 255.0
    g = int(hash_hex[2:4], 16) / 255.0
    b = int(hash_hex[4:6], 16) / 255.0
    
    return (r, g, b)

def blend_colors(color1, color2, ratio=0.5):
    """
    Blend two colors together.
    
    Parameters:
    -----------
    color1 : tuple or str
        First color (RGB tuple or color name)
    color2 : tuple or str
        Second color (RGB tuple or color name)
    ratio : float
        Ratio of color1 to color2 (0.0 = all color2, 1.0 = all color1)
        
    Returns:
    --------
    tuple
        Blended RGB color tuple
    """
    # Convert color names to RGB if needed
    if isinstance(color1, str):
        color1 = matplotlib.colors.to_rgb(color1)
    if isinstance(color2, str):
        color2 = matplotlib.colors.to_rgb(color2)
    
    # Blend the colors
    r = color1[0] * ratio + color2[0] * (1 - ratio)
    g = color1[1] * ratio + color2[1] * (1 - ratio)
    b = color1[2] * ratio + color2[2] * (1 - ratio)
    
    return (r, g, b)

def plot_model_experiments(model_name, all_experiments_data, end_year):
    """
    Create a plot showing all experiments for a specific model.
    
    Parameters:
    -----------
    model_name : str
        Name of the climate model
    all_experiments_data : dict
        Dictionary of experiment data for this model
    end_year : int
        End year for the analysis
    """
    plt.figure(figsize=(12, 8))
    
    # Process each experiment
    for experiment, members in all_experiments_data.items():
        if not members:
            continue
            
        # Prepare data for ensemble statistics
        all_years = set()
        for member_id, member_data in members.items():
            all_years.update(member_data['year'])
        
        all_years = sorted(list(all_years))
        ensemble_data = {year: [] for year in all_years}
        
        # Collect data for each year across all members
        for member_id, member_data in members.items():
            for _, row in member_data.iterrows():
                year = row['year']
                value = row['msftmz']
                ensemble_data[year].append(value)
        
        # Calculate ensemble statistics
        years = []
        means = []
        mins = []
        maxs = []
        
        for year in all_years:
            if ensemble_data[year]:
                years.append(year)
                year_data = np.array(ensemble_data[year])
                means.append(np.mean(year_data))
                mins.append(np.min(year_data))
                maxs.append(np.max(year_data))
        
        # Apply 5-year running mean to ensemble statistics
        if len(years) > 5:
            means_smooth = pd.Series(means).rolling(window=5, center=True).mean().values
            mins_smooth = pd.Series(mins).rolling(window=5, center=True).mean().values
            maxs_smooth = pd.Series(maxs).rolling(window=5, center=True).mean().values
            
            # Create masks for valid values (not NaN)
            means_mask = ~np.isnan(means_smooth)
            mins_mask = ~np.isnan(mins_smooth)
            maxs_mask = ~np.isnan(maxs_smooth)
            
            # Only plot the valid range (excluding first and last 2 years)
            valid_years = np.array(years)[means_mask]
            valid_means = means_smooth[means_mask]
            valid_mins = mins_smooth[mins_mask]
            valid_maxs = maxs_smooth[maxs_mask]
            
            # Plot ensemble mean with thick line
            plt.plot(valid_years, valid_means, linewidth=3, 
                     label=f"{experiment} (n={len(members)})", 
                     color=get_experiment_color(experiment))
            
            # Plot ensemble spread (min to max)
            plt.fill_between(valid_years, valid_mins, valid_maxs, alpha=0.2, 
                            color=get_experiment_color(experiment))
        else:
            # Just one member or too few years, plot it with a thick line
            plt.plot(years, means, linewidth=3, 
                     label=f"{experiment} (n={len(members)})",
                     color=get_experiment_color(experiment))
    
    plt.title(f"AMOC at 26.5°N for {model_name}")
    plt.xlabel("Year")
    plt.ylabel("AMOC (Sv)")
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/{model_name}_amoc_with_spread.png", dpi=300)
    plt.close()

def main(cmip6=False, cmip5=False, end_year=2100, specific_model=None):
    """
    Main function to run the analysis.
    
    Parameters:
    -----------
    cmip6 : bool
        Whether to include CMIP6 models
    cmip5 : bool
        Whether to include CMIP5 models
    end_year : int
        End year for the analysis
    specific_model : str or None
        If provided, only analyze this specific model
    """
    logger.info("Starting main function")
    
    # Define models and experiments
    if cmip6:
        models = CMIP6_MODELS
        experiments = CMIP6_EXPERIMENTS
    elif cmip5:
        models = ALL_CMIP5_MODELS
        experiments = ALL_CMIP5_SCENARIOS
    else:
        logger.error("No model set specified. Use --cmip5 or --cmip6")
        return
    
    # If specific model is provided, filter the list
    if specific_model:
        if specific_model in models:
            models = [specific_model]
        else:
            logger.error(f"Model {specific_model} not found in the selected model set")
            return
    
    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    
    # Dictionary to store all data
    all_data = {model: {} for model in models}
    
    # Process each experiment
    for experiment in experiments:
        logger.info(f"Processing experiment: {experiment}")
        
        # Dictionary to store all model data for this experiment
        all_models_data = {}
        
        # Process each model
        for model_name in models:
            logger.info(f"Processing model: {model_name}")
            
            # Get data for this model and experiment
            model_data = get_model_data(model_name, experiment, end_year)
            
            if model_data:
                all_models_data[model_name] = model_data
                # Store in the all_data dictionary for per-model plots
                all_data[model_name][experiment] = model_data
                logger.info(f"Successfully processed {model_name} for {experiment}")
            else:
                logger.warning(f"No data available for {model_name} - {experiment}")
        
        if not all_models_data:
            logger.warning(f"No data available for any model in {experiment}")
            continue
        
        # Create a plot for this experiment
        plt.figure(figsize=(12, 8))
        
        # Process each model
        for model_name, members in all_models_data.items():
            if members:
                # Prepare data for ensemble statistics
                all_years = set()
                for member_id, member_data in members.items():
                    all_years.update(member_data['year'])
                
                all_years = sorted(list(all_years))
                ensemble_data = {year: [] for year in all_years}
                
                # Collect data for each year across all members
                for member_id, member_data in members.items():
                    for _, row in member_data.iterrows():
                        year = row['year']
                        value = row['msftmz']
                        ensemble_data[year].append(value)
                
                # Calculate ensemble statistics
                years = []
                means = []
                mins = []
                maxs = []
                
                for year in all_years:
                    if ensemble_data[year]:
                        years.append(year)
                        year_data = np.array(ensemble_data[year])
                        means.append(np.mean(year_data))
                        mins.append(np.min(year_data))
                        maxs.append(np.max(year_data))
                
                # Apply 5-year running mean to ensemble statistics
                if len(years) > 5:
                    means_smooth = pd.Series(means).rolling(window=5, center=True).mean().values
                    mins_smooth = pd.Series(mins).rolling(window=5, center=True).mean().values
                    maxs_smooth = pd.Series(maxs).rolling(window=5, center=True).mean().values
                    
                    # Create masks for valid values (not NaN)
                    means_mask = ~np.isnan(means_smooth)
                    mins_mask = ~np.isnan(mins_smooth)
                    maxs_mask = ~np.isnan(maxs_smooth)
                    
                    # Only plot the valid range (excluding first and last 2 years)
                    valid_years = np.array(years)[means_mask]
                    valid_means = means_smooth[means_mask]
                    valid_mins = mins_smooth[mins_mask]
                    valid_maxs = maxs_smooth[maxs_mask]
                    
                    # Plot ensemble mean with thick line
                    plt.plot(valid_years, valid_means, linewidth=3, 
                             label=f"{model_name} (n={len(members)})", 
                             color=get_model_color(model_name))
                    
                    # Plot ensemble spread (min to max)
                    plt.fill_between(valid_years, valid_mins, valid_maxs, alpha=0.2, 
                                    color=get_model_color(model_name))
                else:
                    # Just one member or too few years, plot it with a thick line
                    plt.plot(years, means, linewidth=3, 
                             label=f"{model_name} (n={len(members)})",
                             color=get_model_color(model_name))
        
        plt.title(f"AMOC at 26.5°N for {experiment}")
        plt.xlabel("Year")
        plt.ylabel("AMOC (Sv)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"plots/{experiment}_amoc_with_spread.png", dpi=300)
        plt.close()
    
    # Create per-model plots showing all experiments
    for model_name, experiments_data in all_data.items():
        if experiments_data:
            plot_model_experiments(model_name, experiments_data, end_year)
    
    logger.info("Main function completed")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run AMOC analysis for CMIP models")
    parser.add_argument("--cmip6", action="store_true", help="Use CMIP6 models")
    parser.add_argument("--cmip5", action="store_true", help="Use CMIP5 models")
    parser.add_argument("--end-year", type=int, default=2100, help="End year for the analysis")
    parser.add_argument("--model", type=str, help="Analyze only this specific model")
    
    args = parser.parse_args()
    
    main(cmip6=args.cmip6, cmip5=args.cmip5, end_year=args.end_year, specific_model=args.model) 