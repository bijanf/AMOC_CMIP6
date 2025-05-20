#!/usr/bin/env python3
"""
Script to debug why processing failed for certain CMIP6 models.

Usage:
    python debug_failed_models.py
"""

import sys
import os
import pandas as pd
import traceback

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cmip6_amoc.config import SCENARIOS
from cmip6_amoc.data.freva_finder import find_files_freva
from cmip6_amoc.analysis.amoc import compute_amoc_index
from cmip6_amoc.visualization.plotting import plot_multi_scenario_standardized

# List of models that failed processing
FAILED_MODELS = [
    "UKESM1-0-LL", "GFDL-ESM4", "EC-Earth3", "AWI-CM-1-1-MR", "BCC-CSM2-MR",
    "BCC-ESM1", "CAMS-CSM1-0", "CIESM", "CMCC-CM2-HR4", "CMCC-CM2-SR5",
    "CMCC-ESM2", "CNRM-CM6-1", "CNRM-CM6-1-HR", "CNRM-ESM2-1", "EC-Earth3-AerChem",
    "EC-Earth3-CC", "EC-Earth3-LR", "EC-Earth3-Veg", "EC-Earth3-Veg-LR",
    "FIO-ESM-2-0", "GFDL-CM4", "GFDL-CM4C192", "GISS-E2-1-H", "GISS-E2-2-H",
    "HadGEM3-GC31-LL", "HadGEM3-GC31-MM", "IITM-ESM", "IPSL-CM5A2-INCA",
    "IPSL-CM6A-LR", "IPSL-CM6A-LR-INCA", "KACE-1-0-G", "KIOST-ESM",
    "MCM-UA-1-0", "MIROC-ES2H", "NESM3", "NorESM1-F", "TaiESM1", "UKESM1-1-LL"
]

def debug_model(model_name):
    """Debug why processing failed for a specific model."""
    print(f"\n{'='*80}")
    print(f"Debugging model: {model_name}")
    print(f"{'='*80}")
    
    # Step 1: Check if files exist for each scenario
    scenario_files = {}
    for scenario in SCENARIOS.keys():
        print(f"\nStep 1: Checking for {scenario} files for model {model_name}...")
        try:
            files = find_files_freva(model_name, scenario)
            scenario_files[scenario] = files
            
            if files:
                print(f"  ✓ Found {len(files)} {scenario} files")
                print(f"  First file: {files[0]}")
            else:
                print(f"  ✗ No {scenario} files found")
        except Exception as e:
            print(f"  ✗ Error finding {scenario} files: {e}")
            traceback.print_exc()
    
    # Step 2: Check if any scenario has files
    has_files = any(len(files) > 0 for files in scenario_files.values())
    if not has_files:
        print("\nDiagnosis: Model failed because no files were found for any scenario")
        return
    
    # Step 3: Try to compute AMOC index for each scenario
    scenario_data = {}
    for scenario, files in scenario_files.items():
        if not files:
            continue
            
        print(f"\nStep 3: Computing AMOC index for {scenario}...")
        try:
            df = compute_amoc_index(files, scenario, model_name)
            scenario_data[scenario] = df
            
            if df is not None and not df.empty:
                print(f"  ✓ Successfully computed AMOC index")
                print(f"  DataFrame shape: {df.shape}")
                print(f"  Years: {df.index.min()}-{df.index.max()}")
                print(f"  Ensemble members: {len(df.columns)}")
            else:
                print(f"  ✗ Failed to compute AMOC index (returned empty DataFrame)")
        except Exception as e:
            print(f"  ✗ Error computing AMOC index: {e}")
            traceback.print_exc()
    
    # Step 4: Check if we have data for at least one scenario
    has_data = any(df is not None and not df.empty for df in scenario_data.values())
    if not has_data:
        print("\nDiagnosis: Model failed because no valid AMOC data could be computed")
        return
    
    # Step 5: Try to create the plot
    print("\nStep 5: Creating plot...")
    try:
        plot_multi_scenario_standardized(model_name, scenario_data)
        print("  ✓ Successfully created plot")
    except Exception as e:
        print(f"  ✗ Error creating plot: {e}")
        traceback.print_exc()
    
    # Final diagnosis
    print("\nFinal diagnosis:")
    if not has_files:
        print("  - No files found for any scenario")
    elif not has_data:
        print("  - Files found but no valid AMOC data could be computed")
    else:
        print("  - Files found and AMOC data computed, but plotting failed")
    
    # Recommendations
    print("\nRecommendations:")
    if not has_files:
        print("  - Check if the model name is correct")
        print("  - Check if the model has msftmz data in the CMIP6 archive")
    elif not has_data:
        print("  - Check if the files contain valid msftmz data")
        print("  - Check if the model uses a different variable name for AMOC")
        print("  - Check if the model has a different basin structure")
    else:
        print("  - Check for plotting errors in the log")
        print("  - Try plotting the data manually")

def main():
    """Main function to debug failed models."""
    # Debug each failed model
    for i, model in enumerate(FAILED_MODELS, 1):
        print(f"\nDebugging model {i}/{len(FAILED_MODELS)}: {model}")
        debug_model(model)
    
    print("\nDebugging complete!")

if __name__ == "__main__":
    main() 