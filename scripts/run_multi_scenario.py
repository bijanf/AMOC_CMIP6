#!/usr/bin/env python3
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
    
    print(f"\nUsing model: {model_name}")
    
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
        plot_multi_scenario(model_name, scenario_data)
    else:
        print("No data found for any scenario")

if __name__ == "__main__":
    main()
