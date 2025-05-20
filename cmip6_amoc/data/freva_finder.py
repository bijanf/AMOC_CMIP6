"""Functions for finding CMIP5/CMIP6 data files using freva databrowser."""

import os
import subprocess
from ..config import SCENARIOS

def run_freva_command(command):
    """Run a freva command and return the output.
    
    Args:
        command (str): Freva command to run
        
    Returns:
        str: Command output
    """
    try:
        result = subprocess.run(command, shell=True, check=True, 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               universal_newlines=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running freva command: {e}")
        print(f"Command output: {e.stdout}")
        print(f"Command error: {e.stderr}")
        return ""

def find_models_with_scenarios_freva(scenarios=None, project="CMIP6"):
    """Find models that have data for all specified scenarios using freva.
    
    Args:
        scenarios (list): List of scenario names to check
        project (str): Project name (CMIP5 or CMIP6)
        
    Returns:
        list: Models with data for all specified scenarios
    """
    if scenarios is None:
        if project == "CMIP5":
            scenarios = ["historical", "rcp85"]
        else:  # CMIP6
            scenarios = ["historical", "ssp585"]
    
    models_by_scenario = {}
    
    for scenario in scenarios:
        # Construct freva databrowser command with correct syntax
        freva_cmd = f'freva databrowser project={project} experiment={scenario} variable=msftmz'
        
        # Run the command
        output = run_freva_command(freva_cmd)
        
        if not output:
            print(f"No output from freva for scenario {scenario} in {project}")
            continue
        
        # Parse the output (list of file paths)
        file_paths = [line.strip() for line in output.strip().split('\n') if line.strip()]
        
        # Extract model names from file paths
        models = set()
        for path in file_paths:
            # Extract model name from file path or filename
            filename = os.path.basename(path)
            if "_Omon_" in filename:
                parts = filename.split("_Omon_")[1].split("_")
                if parts:
                    models.add(parts[0])
        
        models_by_scenario[scenario] = models
        print(f"Found {len(models)} models for {scenario} in {project} using freva")
    
    # Find models that have data for all scenarios
    if not models_by_scenario:
        return []
    
    # Start with models from first scenario
    first_scenario = scenarios[0]
    common_models = models_by_scenario.get(first_scenario, set())
    
    # Intersect with models from other scenarios
    for scenario in scenarios[1:]:
        common_models &= models_by_scenario.get(scenario, set())
    
    return sorted(common_models)

def find_files_freva(model_name, scenario, project="CMIP6"):
    """Find CMIP5/CMIP6 files for a specific model and scenario using freva.
    
    Args:
        model_name (str): Name of the CMIP5/CMIP6 model
        scenario (str): Name of the scenario/experiment
        project (str): Project name (CMIP5 or CMIP6)
        
    Returns:
        list: Paths to matching netCDF files
    """
    # Construct freva databrowser command with correct syntax
    freva_cmd = f'freva databrowser project={project} model={model_name} experiment={scenario} variable=msftmz'
    
    # Run the command
    output = run_freva_command(freva_cmd)
    
    if not output:
        print(f"No output from freva for model {model_name}, scenario {scenario} in {project}")
        return []
    
    # Parse the output (list of file paths)
    file_paths = [line.strip() for line in output.strip().split('\n') if line.strip()]
    
    print(f"Found {len(file_paths)} files for {model_name}, {scenario} in {project}")
    return file_paths 