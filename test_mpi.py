#!/usr/bin/env python3

import sys
import importlib.util

# Load the run_all_models_freva module
spec = importlib.util.spec_from_file_location("run_all_models_freva", "scripts/run_all_models_freva.py")
run_all_models_freva = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_all_models_freva)

# Override the CMIP6_MODELS list
run_all_models_freva.CMIP6_MODELS = ["MPI-ESM1-2-LR"]

# Run the main function
run_all_models_freva.main(cmip6=True, cmip5=False, end_year=2100)