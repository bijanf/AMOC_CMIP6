"""Statistical analysis functions for AMOC data."""

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
