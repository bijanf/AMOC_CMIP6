"""Functions for visualizing AMOC data."""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ..config import SCENARIOS, PLOT_DIR
from ..analysis.statistics import calculate_ensemble_statistics
import os
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
import matplotlib.cm as cm

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
        'lower_25': df.quantile(0.25, axis=1),
        'upper_75': df.quantile(0.75, axis=1),
        'n_members': len(df.columns)
    }
    
    return stats

def plot_multi_scenario(model_name, scenario_data, y_limits=None):
    """Create a multi-scenario AMOC plot.
    
    Args:
        model_name (str): Name of the model
        scenario_data (dict): Dictionary mapping scenario names to DataFrames
        y_limits (tuple): Optional (min, max) for y-axis
        
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
    
    # Set y-axis limits if provided
    if y_limits:
        plt.ylim(y_limits)
    
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

def plot_multi_scenario_standardized(model_name, scenario_data, end_year=2100):
    """Create a multi-scenario plot with standardized y-axis scale.
    
    Args:
        model_name (str): Name of the model
        scenario_data (dict): Dictionary mapping scenario names to DataFrames
        end_year (int, optional): End year for the plot. Defaults to 2100.
    """
    # Create plot directory if it doesn't exist
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # Set up the figure
    plt.figure(figsize=(12, 8))
    
    # Set up colors and line styles
    colors = {
        'historical': 'black',
        'ssp119': '#1a9850',
        'ssp126': '#66bd63',
        'ssp245': '#fdae61',
        'ssp370': '#f46d43',
        'ssp434': '#d73027',
        'ssp460': '#a50026',
        'ssp534-over': '#7a0177',
        'ssp585': '#d73027',
        'rcp26': '#66bd63',
        'rcp45': '#fdae61',
        'rcp60': '#f46d43',
        'rcp85': '#d73027'
    }
    
    # Track min and max values for y-axis scaling
    all_values = []
    
    # Plot each scenario
    for scenario, df in scenario_data.items():
        # Filter data to end_year
        if end_year is not None:
            df = df[df.index <= end_year]
        
        if df.empty:
            print(f"Warning: No data for {scenario} within the specified time range")
            continue
            
        # Get number of ensemble members (columns)
        num_members = df.shape[1]
            
        # Calculate ensemble mean
        mean_series = df.mean(axis=1)
        
        # Calculate ensemble spread (min to max)
        min_series = df.min(axis=1)
        max_series = df.max(axis=1)
        
        # Get color for this scenario
        color = colors.get(scenario, 'gray')
        
        # Plot ensemble mean with number of members in the label
        plt.plot(mean_series.index, mean_series.values, 
                 label=f"{SCENARIOS.get(scenario, scenario)} (n={num_members})", 
                 color=color, linewidth=2)
        
        # Plot ensemble spread as shaded area
        plt.fill_between(min_series.index, min_series.values, max_series.values, 
                         color=color, alpha=0.2)
        
        # Collect values for y-axis scaling
        all_values.extend(mean_series.values)
        all_values.extend(min_series.values)
        all_values.extend(max_series.values)
    
    # Set up plot labels and title
    plt.title(f"AMOC Strength at 26.5°N - {model_name}", fontsize=16)
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("AMOC Strength (Sv)", fontsize=14)
    
    # Set up x-axis
    plt.xlim(1850, end_year)
    plt.grid(True, alpha=0.3)
    
    # Set up y-axis with standardized scale
    if all_values:
        data_min = np.nanmin(all_values)
        data_max = np.nanmax(all_values)
        
        # Set y-axis limits with some padding
        y_range = data_max - data_min
        y_min = max(0, data_min - 0.1 * y_range)  # Don't go below 0
        y_max = data_max + 0.1 * y_range
        
        plt.ylim(y_min, y_max)
    
    # Add legend
    plt.legend(loc='best', fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    filename = os.path.join(PLOT_DIR, f"{model_name.replace(' ', '_')}_amoc_multi_scenario.png")
    plt.savefig(filename, dpi=300)
    plt.close()
    
    print(f"Plot saved as {filename}")

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

def plot_multi_model_comparison(models_data, scenario, end_year=2100):
    """Create a multi-model comparison plot for a specific scenario.
    
    Args:
        models_data (dict): Dictionary mapping model names to DataFrames
        scenario (str): Scenario name
        end_year (int, optional): End year for the plot. Defaults to 2100.
    """
    # Create plot directory if it doesn't exist
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # Set up the figure
    plt.figure(figsize=(12, 8))
    
    # Set up colors using a colormap instead of seaborn
    cmap = cm.get_cmap('tab20')
    num_models = len(models_data)
    
    # Track min and max values for y-axis scaling
    all_values = []
    
    # Plot each model
    for i, (model_name, df) in enumerate(models_data.items()):
        # Filter data to end_year
        if end_year is not None:
            df = df[df.index <= end_year]
            
        if df.empty:
            print(f"Warning: No data for {model_name} within the specified time range")
            continue
            
        # Get number of ensemble members (columns)
        num_members = df.shape[1]
            
        # Calculate ensemble mean
        mean_series = df.mean(axis=1)
        
        # Get color for this model
        color = cmap(i % 20)  # Use modulo to cycle through colors if more than 20 models
        
        # Plot ensemble mean with number of members in the label
        plt.plot(mean_series.index, mean_series.values, 
                 label=f"{model_name} (n={num_members})", 
                 color=color, linewidth=2)
        
        # Collect values for y-axis scaling
        all_values.extend(mean_series.values)
    
    # Set up plot labels and title
    scenario_name = SCENARIOS.get(scenario, scenario)
    plt.title(f"AMOC Strength at 26.5°N - {scenario_name} Scenario", fontsize=16)
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("AMOC Strength (Sv)", fontsize=14)
    
    # Set up x-axis
    plt.xlim(1850, end_year)
    plt.grid(True, alpha=0.3)
    
    # Set up y-axis with standardized scale
    if all_values:
        data_min = np.nanmin(all_values)
        data_max = np.nanmax(all_values)
        
        # Set y-axis limits with some padding
        y_range = data_max - data_min
        y_min = max(0, data_min - 0.1 * y_range)  # Don't go below 0
        y_max = data_max + 0.1 * y_range
        
        plt.ylim(y_min, y_max)
    
    # Add legend
    plt.legend(loc='best', fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    filename = os.path.join(PLOT_DIR, f"multi_model_{scenario}_comparison.png")
    plt.savefig(filename, dpi=300)
    plt.close()
    
    print(f"Plot saved as {filename}")

def plot_amoc_percentage_changes(model_data, time_slices=None):
    """Create a single plot showing percentage changes in AMOC for all models.
    
    Args:
        model_data (dict): Dictionary containing model data with scenarios
        time_slices (list): List of tuples defining time slices [(start, end), ...]
    """
    if time_slices is None:
        time_slices = [
            (1981, 2010),  # Historical baseline
            (2011, 2040),  # Near future
            (2041, 2070),  # Mid future
            (2071, 2100)   # Far future
        ]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Calculate percentage changes for each model and scenario
    results = {}
    for model_name, scenarios in model_data.items():
        results[model_name] = {}
        for scenario, df in scenarios.items():
            if scenario == 'historical':
                continue
                
            # Calculate baseline mean (1981-2010)
            baseline = df.loc[1981:2010].mean()
            
            # Calculate percentage changes for each time slice
            changes = []
            for start, end in time_slices:
                if start == 1981:  # Skip baseline period
                    changes.append(0)
                    continue
                    
                future_mean = df.loc[start:end].mean()
                percent_change = ((future_mean - baseline) / baseline) * 100
                changes.append(percent_change)
            
            results[model_name][scenario] = changes
    
    # Create plot
    x = np.arange(len(time_slices))
    width = 0.8 / len(results)  # Adjust bar width based on number of models
    
    # Define colors for scenarios
    scenario_colors = {
        'ssp126': 'blue',
        'ssp245': 'orange',
        'ssp370': 'red',
        'ssp585': 'darkred'
    }
    
    # Plot bars for each model
    for i, (model_name, scenario_changes) in enumerate(results.items()):
        for scenario, changes in scenario_changes.items():
            if scenario in scenario_colors:
                offset = (i - len(results)/2 + 0.5) * width
                plt.bar(x + offset, changes, width, 
                       label=f"{model_name} - {scenario}",
                       color=scenario_colors[scenario],
                       alpha=0.7)
    
    # Customize plot
    plt.xlabel('Time Period', fontsize=12)
    plt.ylabel('AMOC Change (%)', fontsize=12)
    plt.title('AMOC Weakening Across CMIP5/CMIP6 Models\nRelative to 1981-2010 Baseline', fontsize=14)
    
    # Set x-axis labels
    plt.xticks(x, [f'{start}-{end}' for start, end in time_slices], rotation=45)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add legend with model and scenario information
    handles, labels = plt.gca().get_legend_handles_labels()
    # Sort legend entries by scenario
    sorted_pairs = sorted(zip(handles, labels), key=lambda x: x[1])
    handles, labels = zip(*sorted_pairs)
    plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add zero line
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    filename = os.path.join(PLOT_DIR, "amoc_percentage_changes_combined.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved as {filename}")
