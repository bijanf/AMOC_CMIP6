#!/usr/bin/env python3
"""
Script to analyze AMOC across multiple scenarios for all available models.

Usage:
    python run_all_models.py
"""

import sys
import os
import pandas as pd
import concurrent.futures
import glob

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cmip6_amoc.config import SCENARIOS, RECOMMENDED_MODELS, CMIP6_DATA_PATH
from cmip6_amoc.data.finder import find_models_with_scenarios, find_files
from cmip6_amoc.analysis.amoc import compute_amoc_index
from cmip6_amoc.visualization.plotting import plot_multi_scenario_standardized

def process_model(model_name):
    """Process a single model and create plots."""
    print(f"\n{'='*50}")
    print(f"Processing model: {model_name}")
    print(f"{'='*50}")
    
    # Process each scenario
    scenario_data = {}
    for scenario in SCENARIOS.keys():
        print(f"\nSearching for {scenario} files for model {model_name}...")
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

def check_model_availability():
    """Check which models are actually available in the data directory."""
    print("\nChecking actual model availability in data directory...")
    
    # Define a function to search for msftmz files with flexible patterns
    def find_msftmz_files(scenario):
        # Start with the base CMIP6 data path
        base_path = CMIP6_DATA_PATH
        
        # Try different directory structures
        patterns = [
            # Standard CMIP6 structure
            os.path.join(base_path, "*", "*", "*", scenario, "*", "Omon", "msftmz", "*", "*.nc"),
            # Alternative structure without version directory
            os.path.join(base_path, "*", "*", "*", scenario, "*", "Omon", "msftmz", "*.nc"),
            # Try with explicit activity and institution
            os.path.join(base_path, "CMIP", "*", "*", scenario, "*", "Omon", "msftmz", "*", "*.nc"),
            os.path.join(base_path, "ScenarioMIP", "*", "*", scenario, "*", "Omon", "msftmz", "*", "*.nc"),
            # Look in subdirectories
            os.path.join(base_path, "*", "*", "*", "*", scenario, "*", "Omon", "msftmz", "*", "*.nc"),
            # Try parent directory
            os.path.join(os.path.dirname(base_path), "*", "*", "*", "*", scenario, "*", "Omon", "msftmz", "*", "*.nc"),
        ]
        
        # Collect all files matching any pattern
        all_files = []
        for pattern in patterns:
            files = glob.glob(pattern)
            all_files.extend(files)
            if files:
                print(f"Found {len(files)} files with pattern: {pattern}")
        
        return all_files
    
    # Find files for historical and all SSP scenarios
    historical_files = find_msftmz_files("historical")
    
    # Dictionary to store files for each SSP scenario
    ssp_files = {}
    for scenario in ["ssp126", "ssp245", "ssp370", "ssp585"]:
        ssp_files[scenario] = find_msftmz_files(scenario)
    
    print(f"\nFound {len(historical_files)} historical files")
    for scenario, files in ssp_files.items():
        print(f"Found {len(files)} {scenario} files")
    
    # Extract model names from filenames
    def extract_models_from_files(files):
        models = set()
        for file in files:
            filename = os.path.basename(file)
            if "_Omon_" in filename:
                parts = filename.split("_Omon_")[1].split("_")
                if parts:
                    models.add(parts[0])
        return models
    
    historical_models = extract_models_from_files(historical_files)
    
    # Dictionary to store models for each SSP scenario
    ssp_models = {}
    for scenario, files in ssp_files.items():
        ssp_models[scenario] = extract_models_from_files(files)
    
    # Find models with historical and at least one SSP scenario
    models_with_historical_and_any_ssp = set()
    for model in historical_models:
        for scenario, models in ssp_models.items():
            if model in models:
                models_with_historical_and_any_ssp.add(model)
                break
    
    # Find models with all scenarios
    models_with_all_scenarios = historical_models.copy()
    for scenario, models in ssp_models.items():
        models_with_all_scenarios &= models
    
    # Print results
    print(f"\nFound {len(historical_models)} models with historical data:")
    for model in sorted(historical_models):
        print(f"  - {model}")
    
    for scenario, models in ssp_models.items():
        print(f"\nFound {len(models)} models with {scenario} data:")
        for model in sorted(models):
            print(f"  - {model}")
    
    print(f"\nFound {len(models_with_historical_and_any_ssp)} models with historical and at least one SSP scenario:")
    for model in sorted(models_with_historical_and_any_ssp):
        print(f"  - {model}")
    
    print(f"\nFound {len(models_with_all_scenarios)} models with all scenarios:")
    for model in sorted(models_with_all_scenarios):
        print(f"  - {model}")
    
    # Return models with historical and at least SSP585
    models_with_historical_and_ssp585 = historical_models & ssp_models.get("ssp585", set())
    print(f"\nFound {len(models_with_historical_and_ssp585)} models with both historical and SSP585 data:")
    for model in sorted(models_with_historical_and_ssp585):
        print(f"  - {model}")
    
    return sorted(models_with_historical_and_ssp585)

def main():
    """Main function to run AMOC analysis for all models."""
    # First, check which models are actually available
    available_models = check_model_availability()
    
    if not available_models:
        print("No models found with required scenarios")
        return
    
    # Add any specific models that might be missing
    additional_models = [
        "ACCESS-CM2", "ACCESS-ESM1-5", "AWI-CM-1-1-MR", "BCC-CSM2-MR",
        "CAMS-CSM1-0", "CNRM-CM6-1", "CNRM-ESM2-1", "E3SM-1-0",
        "FGOALS-f3-L", "FGOALS-g3", "FIO-ESM-2-0", "GISS-E2-1-G",
        "HadGEM3-GC31-LL", "HadGEM3-GC31-MM", "INM-CM4-8", "INM-CM5-0",
        "IPSL-CM6A-LR", "KACE-1-0-G", "KIOST-ESM", "MCM-UA-1-0",
        "MIROC-ES2L", "MPI-ESM-1-2-HAM", "MRI-ESM2-0", "NESM3",
        "NorCPM1", "SAM0-UNICON", "TaiESM1"
    ]
    
    for model in additional_models:
        if model not in available_models:
            # Check if this model has files
            historical_files = find_files(model, "historical")
            ssp585_files = find_files(model, "ssp585")
            
            if historical_files and ssp585_files:
                available_models.append(model)
                print(f"Added model {model} to processing list")
    
    print(f"\nWill process {len(available_models)} models:")
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
    print("\n" + "="*50)
    print("Processing Summary:")
    print("="*50)
    successful = [model for model, success in results.items() if success]
    failed = [model for model, success in results.items() if not success]
    
    print(f"Successfully processed {len(successful)} models:")
    for model in successful:
        print(f"  - {model}")
    
    if failed:
        print(f"\nFailed to process {len(failed)} models:")
        for model in failed:
            print(f"  - {model}")
    
    print("\nAll plots have been created with standardized y-axis scales.")

if __name__ == "__main__":
    main()
