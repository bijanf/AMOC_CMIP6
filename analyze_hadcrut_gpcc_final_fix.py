#!/usr/bin/env python3
"""FINAL CORRECTED version: Analyze temperature trends using HadCRUT data and precipitation trends using GPCC data for Central Asia regions."""

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from datetime import datetime
from scipy import stats

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'temp_precip_final_fix_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
START_YEAR = 1950
END_YEAR = 2019
OUTPUT_DIR = Path('output')
PLOTS_DIR = Path('plots')

# Create output directories
OUTPUT_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# Define regions of interest - EXPANDED boundaries to capture more grid points
REGIONS = {
    'Aral_Sea': {'lat': (42, 48), 'lon': (57, 63)},
    'Kyrgyzstan': {'lat': (39, 43), 'lon': (70, 80)},
    'Kazakhstan': {'lat': (44, 54), 'lon': (58, 77)},
    'Tajikistan': {'lat': (36, 40), 'lon': (67, 75)},
    'Uzbekistan': {'lat': (37, 45), 'lon': (56, 67)},
    'Turkmenistan': {'lat': (35, 42), 'lon': (52, 64)}
}

# Data file paths
HADCRUT_FILE = "/work/bm1159/XCES/data4xces/raw2cmor/obs4mips/DRS/observations/grid/MOHC-CRU/MOHC-CRU/HADCRUT4-MEDIAN/mon/atmos/mon/r1i1p1/v20190209/tas/tas_mon_MOHC-CRU_HADCRUT4-MEDIAN_r1i1p1_185001-201812.nc"
GPCC_FILE = "/work/bm1159/XCES/data4xces/raw2cmor/obs4mips/DRS/observations/grid/DWD/DWD/GPCC/mon/atmos/mon/r1i1p1/v20160809/pr/pr_mon_DWD_GPCC_r1i1p1_190101-201312.nc"

def process_temperature_data():
    """Process HadCRUT temperature data for regional analysis."""
    logger.info("Processing HadCRUT temperature data...")
    
    try:
        if not Path(HADCRUT_FILE).exists():
            logger.error(f"HadCRUT file not found: {HADCRUT_FILE}")
            return None
        
        with xr.open_dataset(HADCRUT_FILE, use_cftime=True) as ds:
            logger.info("HadCRUT dataset loaded successfully")
            
            temp_data = ds['tas']
            logger.info(f"Temperature data shape: {temp_data.shape}")
            
            units = temp_data.attrs.get('units', 'Unknown')
            logger.info(f"Temperature units: {units}")
            
            # Filter time range
            time_mask = (temp_data.time.dt.year >= START_YEAR) & (temp_data.time.dt.year <= END_YEAR)
            temp_data = temp_data.sel(time=time_mask)
            logger.info(f"Filtered to {START_YEAR}-{END_YEAR}: {temp_data.shape}")
            
            regional_trends = {}
            
            for region_name, bounds in REGIONS.items():
                logger.info(f"Processing temperature for region: {region_name}")
                
                try:
                    lat_min, lat_max = bounds['lat']
                    lon_min, lon_max = bounds['lon']
                    
                    # Select region
                    regional_temp = temp_data.sel(
                        lat=slice(lat_min, lat_max),
                        lon=slice(lon_min, lon_max)
                    )
                    
                    logger.info(f"Regional temperature data shape for {region_name}: {regional_temp.shape}")
                    
                    # Calculate spatial mean
                    regional_mean = regional_temp.mean(dim=['lat', 'lon'])
                    
                    # FIXED: HadCRUT contains temperature ANOMALIES, not absolute temperatures
                    # The data is already in the correct units (K anomalies = °C anomalies)
                    # Just mask extreme outliers (anomalies should be reasonable)
                    regional_mean = regional_mean.where((regional_mean > -50) & (regional_mean < 50))
                    logger.info(f"Temperature anomalies processed for {region_name} (K anomalies = °C anomalies)")
                    
                    # Check data range
                    valid_data = regional_mean.dropna(dim='time')
                    if len(valid_data) < 10:
                        logger.warning(f"Not enough valid temperature data for {region_name}")
                        continue
                    
                    data_min = float(valid_data.min())
                    data_max = float(valid_data.max())
                    data_mean = float(valid_data.mean())
                    logger.info(f"Temperature range for {region_name}: {data_min:.1f} to {data_max:.1f}°C, mean: {data_mean:.1f}°C")
                    
                    # Convert to annual means
                    annual_temp = regional_mean.groupby('time.year').mean()
                    logger.info(f"Annual temperature data for {region_name}: shape {annual_temp.shape}")
                    
                    # Calculate trend
                    years = annual_temp.year.values
                    values = annual_temp.values
                    
                    # Remove NaN values
                    valid_mask = ~np.isnan(values)
                    if np.sum(valid_mask) < 10:
                        logger.warning(f"Not enough valid temperature data for {region_name}")
                        continue
                    
                    years_valid = years[valid_mask]
                    values_valid = values[valid_mask]
                    
                    logger.info(f"Valid temperature data points for {region_name}: {len(years_valid)}")
                    logger.info(f"Temperature range: {values_valid.min():.1f} to {values_valid.max():.1f}°C")
                    
                    # Calculate linear trend
                    slope, intercept, r_value, p_value, std_err = stats.linregress(years_valid, values_valid)
                    trend_per_decade = slope * 10
                    
                    logger.info(f"Temperature trend for {region_name}: {trend_per_decade:.2f}°C/decade (p={p_value:.3f})")
                    
                    regional_trends[region_name] = {
                        'mean_trend': trend_per_decade,
                        'p_value': p_value,
                        'r_value': r_value,
                        'significant': p_value < 0.05,
                        'mean_value': float(np.mean(values_valid))
                    }
                    
                except Exception as e:
                    logger.error(f"Error processing temperature for {region_name}: {str(e)}")
                    continue
            
            return regional_trends
            
    except Exception as e:
        logger.error(f"Error loading HadCRUT data: {str(e)}")
        return None

def process_precipitation_data():
    """Process GPCC precipitation data for regional analysis."""
    logger.info("Processing GPCC precipitation data...")
    
    try:
        # Check if file exists
        if not Path(GPCC_FILE).exists():
            logger.error(f"GPCC file not found: {GPCC_FILE}")
            return None
        
        # Load GPCC data
        with xr.open_dataset(GPCC_FILE, use_cftime=True) as ds:
            logger.info(f"GPCC dataset loaded successfully")
            logger.info(f"Precipitation data shape: {ds.pr.shape}")
            logger.info(f"Precipitation units (metadata): {ds.pr.attrs.get('units', 'Unknown')}")
            logger.info("Note: GPCC units metadata is incorrect - data is actually in kg/m²/s")
            
            # Filter time range
            time_mask = (ds.time.dt.year >= START_YEAR) & (ds.time.dt.year <= END_YEAR)
            precip_data = ds.pr.sel(time=time_mask)
            logger.info(f"Filtered to {START_YEAR}-{END_YEAR}: {precip_data.shape}")
            
            # Process each region
            regional_trends = {}
            
            for region_name, bounds in REGIONS.items():
                logger.info(f"Processing precipitation for region: {region_name}")
                
                try:
                    # Extract regional data
                    lat_min, lat_max = bounds['lat']
                    lon_min, lon_max = bounds['lon']
                    
                    # Select region
                    regional_precip = precip_data.sel(
                        lat=slice(lat_min, lat_max),
                        lon=slice(lon_min, lon_max)
                    )
                    
                    logger.info(f"Regional precipitation data shape for {region_name}: {regional_precip.shape}")
                    
                    # Calculate spatial mean for the region
                    regional_mean = regional_precip.mean(dim=['lat', 'lon'])
                    
                    # GPCC data is in kg/m²/s, convert to mm/month
                    # 1 kg/m²/s = 86400 mm/day = 86400 * 30.44 mm/month (average month length)
                    regional_mean_mm_month = regional_mean * 86400 * 30.44
                    logger.info(f"Converted from kg/m²/s to mm/month for {region_name}")
                    
                    # Check sample values
                    sample_values = regional_mean_mm_month.isel(time=slice(0, 5)).values
                    logger.info(f"Sample monthly values for {region_name}: {sample_values}")
                    
                    # Convert to annual totals (sum monthly values)
                    annual_precip = regional_mean_mm_month.groupby('time.year').sum()
                    logger.info(f"Annual precipitation data for {region_name}: shape {annual_precip.shape}")
                    
                    # Calculate statistics
                    years = annual_precip.year.values
                    values = annual_precip.values
                    
                    # Remove NaN values and check for reasonable precipitation values
                    # Reasonable range for annual precipitation: 0 to 5000 mm/year
                    valid_mask = ~np.isnan(values) & (values >= 0) & (values < 5000)
                    if np.sum(valid_mask) < 10:
                        logger.warning(f"Not enough valid precipitation data for {region_name}")
                        logger.warning(f"Values range: {values.min():.1f} to {values.max():.1f} mm/year")
                        continue
                    
                    years_valid = years[valid_mask]
                    values_valid = values[valid_mask]
                    
                    mean_annual = float(np.mean(values_valid))
                    logger.info(f"Mean annual precipitation for {region_name}: {mean_annual:.1f} mm/year")
                    logger.info(f"Valid precipitation data points for {region_name}: {len(years_valid)}")
                    logger.info(f"Precipitation range: {values_valid.min():.1f} to {values_valid.max():.1f} mm/year")
                    
                    # Calculate linear trend
                    slope, intercept, r_value, p_value, std_err = stats.linregress(years_valid, values_valid)
                    trend_per_decade = slope * 10
                    
                    logger.info(f"Precipitation trend for {region_name}: {trend_per_decade:.1f} mm/year per decade (p={p_value:.3f})")
                    
                    regional_trends[region_name] = {
                        'mean_trend': trend_per_decade,
                        'p_value': p_value,
                        'r_value': r_value,
                        'significant': p_value < 0.05,
                        'mean_value': mean_annual
                    }
                    
                except Exception as e:
                    logger.error(f"Error processing precipitation for {region_name}: {str(e)}")
                    continue
            
            return regional_trends
            
    except Exception as e:
        logger.error(f"Error loading GPCC data: {str(e)}")
        return None

def create_summary_and_plots(temp_trends, precip_trends):
    """Create summary files and plots."""
    
    # Temperature summary
    if temp_trends:
        logger.info(f"Temperature trends calculated for {len(temp_trends)} regions")
        
        # Create summary DataFrame
        temp_summary = []
        for region, stats in temp_trends.items():
            temp_summary.append({
                'Region': region,
                'Trend_per_decade': f"{stats['mean_trend']:.3f}",
                'Mean_Temperature': f"{stats['mean_value']:.1f}",
                'P_value': f"{stats['p_value']:.3f}",
                'Significant': 'Yes' if stats['significant'] else 'No',
                'R_value': f"{stats['r_value']:.3f}"
            })
        
        temp_df = pd.DataFrame(temp_summary)
        temp_file = OUTPUT_DIR / f"hadcrut_temperature_trends_final_{START_YEAR}_{END_YEAR}.csv"
        temp_df.to_csv(temp_file, index=False)
        logger.info(f"Temperature summary saved: {temp_file}")
        
        # Temperature plot
        fig, ax = plt.subplots(figsize=(12, 8))
        regions = list(temp_trends.keys())
        trends = [temp_trends[region]['mean_trend'] for region in regions]
        p_values = [temp_trends[region]['p_value'] for region in regions]
        
        bars = ax.bar(regions, trends)
        for i, (bar, p_val) in enumerate(zip(bars, p_values)):
            if p_val < 0.05:
                bar.set_color('red' if trends[i] > 0 else 'blue')
            else:
                bar.set_color('lightgray')
        
        ax.set_ylabel('Temperature Trend (°C/decade)')
        ax.set_title(f'HadCRUT Temperature Trends ({START_YEAR}-{END_YEAR})')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plot_file = PLOTS_DIR / f"hadcrut_temperature_trends_final_{START_YEAR}_{END_YEAR}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Temperature plot saved: {plot_file}")
    
    # Precipitation summary
    if precip_trends:
        logger.info(f"Precipitation trends calculated for {len(precip_trends)} regions")
        
        # Create summary DataFrame
        precip_summary = []
        for region, stats in precip_trends.items():
            precip_summary.append({
                'Region': region,
                'Trend_per_decade': f"{stats['mean_trend']:.1f}",
                'Mean_Precipitation': f"{stats['mean_value']:.1f}",
                'P_value': f"{stats['p_value']:.3f}",
                'Significant': 'Yes' if stats['significant'] else 'No',
                'R_value': f"{stats['r_value']:.3f}"
            })
        
        precip_df = pd.DataFrame(precip_summary)
        precip_file = OUTPUT_DIR / f"gpcc_precipitation_trends_final_{START_YEAR}_{END_YEAR}.csv"
        precip_df.to_csv(precip_file, index=False)
        logger.info(f"Precipitation summary saved: {precip_file}")
        
        # Precipitation plot
        fig, ax = plt.subplots(figsize=(12, 8))
        regions = list(precip_trends.keys())
        trends = [precip_trends[region]['mean_trend'] for region in regions]
        p_values = [precip_trends[region]['p_value'] for region in regions]
        
        bars = ax.bar(regions, trends)
        for i, (bar, p_val) in enumerate(zip(bars, p_values)):
            if p_val < 0.05:
                bar.set_color('blue' if trends[i] > 0 else 'brown')
            else:
                bar.set_color('lightgray')
        
        ax.set_ylabel('Precipitation Trend (mm/year/decade)')
        ax.set_title(f'GPCC Precipitation Trends ({START_YEAR}-{END_YEAR})')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plot_file = PLOTS_DIR / f"gpcc_precipitation_trends_final_{START_YEAR}_{END_YEAR}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Precipitation plot saved: {plot_file}")

def main():
    """Main function."""
    logger.info("Starting FINAL CORRECTED HadCRUT temperature and GPCC precipitation analysis...")
    
    # Process temperature data
    temp_trends = process_temperature_data()
    if not temp_trends:
        logger.warning("No temperature trends calculated")
    
    # Process precipitation data
    precip_trends = process_precipitation_data()
    if not precip_trends:
        logger.warning("No precipitation trends calculated")
    
    # Create summaries and plots
    if temp_trends or precip_trends:
        create_summary_and_plots(temp_trends, precip_trends)
        
        # Print summary
        print("\n" + "="*60)
        print("FINAL ANALYSIS RESULTS SUMMARY")
        print("="*60)
        
        if temp_trends:
            print("\nTEMPERATURE TRENDS (°C/decade):")
            print("-"*40)
            for region, stats in temp_trends.items():
                significance = "*" if stats['significant'] else " "
                print(f"{region:12s}: {stats['mean_trend']:+6.3f}°C/decade{significance}")
                print(f"              (Mean: {stats['mean_value']:.1f}°C)")
        
        if precip_trends:
            print("\nPRECIPITATION TRENDS (mm/year/decade):")
            print("-"*40)
            for region, stats in precip_trends.items():
                significance = "*" if stats['significant'] else " "
                print(f"{region:12s}: {stats['mean_trend']:+7.1f} mm/year/decade{significance}")
                print(f"              (Mean annual: {stats['mean_value']:.1f} mm/year)")
        
        print("\n* = statistically significant (p < 0.05)")
    else:
        logger.error("No trends calculated for either temperature or precipitation")
    
    logger.info("FINAL CORRECTED analysis complete!")

if __name__ == "__main__":
    main() 