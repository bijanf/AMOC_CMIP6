"""Functions for finding CMIP6 data files."""

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
    
    # Define multiple search patterns to be more flexible
    patterns = []
    
    # Standard pattern with path component
    path_component = SCENARIOS[scenario]["path_component"]
    patterns.append(os.path.join(
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
    ))
    
    # Alternative pattern without version directory
    patterns.append(os.path.join(
        CMIP6_DATA_PATH, 
        path_component, 
        "*", 
        model_name, 
        scenario, 
        "*", 
        "Omon", 
        "msftmz", 
        "*", 
        f"msftmz_Omon_{model_name}_{scenario}_*.nc"
    ))
    
    # More flexible pattern
    patterns.append(os.path.join(
        CMIP6_DATA_PATH, 
        "*", 
        "*", 
        model_name, 
        scenario, 
        "*", 
        "Omon", 
        "msftmz", 
        "*", 
        "*.nc"
    ))
    
    # Try parent directory
    patterns.append(os.path.join(
        os.path.dirname(CMIP6_DATA_PATH), 
        "*", 
        "*", 
        "*", 
        model_name, 
        scenario, 
        "*", 
        "Omon", 
        "msftmz", 
        "*", 
        "*.nc"
    ))
    
    # Collect all files matching any pattern
    all_files = []
    for pattern in patterns:
        files = glob.glob(pattern)
        if files:
            print(f"Found {len(files)} files with pattern: {pattern}")
            all_files.extend(files)
    
    return all_files
