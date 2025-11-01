import pandas as pd
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import date, timedelta, datetime

def resample_to_hourly(df, value_col):
        """Resample internal sensor data to hourly timesteps."""
        if df.empty:
            return df
        
        # Ensure sorted index
        df = df.sort_values("datetime").set_index("datetime")
        
        # Resample to hourly with mean; forward fill occasional gaps
        hourly = df[value_col].resample("1h").mean()
        hourly = hourly.ffill(limit=2)  # Fill gaps up to 2 hours
        
        return hourly.to_frame(value_col)

def create_hourly_forecast(df_forecast, df_weather):
    forecast_data = pd.concat([df_weather, df_forecast], ignore_index=True)
    #To compensate the addition of the forecasted day, the first day of the dataset is dropped
    forecast_data = forecast_data[pd.to_datetime(forecast_data["datetime"], errors="coerce").notna()]
    forecast_data["datetime"] = pd.to_datetime(forecast_data["datetime"], errors="coerce")
    first_day = forecast_data["datetime"].dt.date.min()
    forecast_data = forecast_data[forecast_data["datetime"].dt.date != first_day]
    forecast_data.reset_index(drop=True, inplace=True)
    forecast_data["datetime"] = forecast_data["datetime"] - timedelta(days=1)
    rename_dict = {
        'temp_ext': 'temp_forecast',
        'humid_ext': 'humid_forecast',
        'sunshine_duration': 'sunshine_duration_forecast'
    }

    return forecast_data.rename(columns=rename_dict)

def join_hourly_data(temp_data, humid_data, weather_data, forecast_data):
    """Join all hourly data sources."""
    
    # Set weather data index
    weather_data = weather_data.sort_values("datetime").set_index("datetime")

    #Set forecasting data
    forecast_data = forecast_data.sort_values("datetime").set_index("datetime")
    
    # Join all data sources
    joined = weather_data.join(temp_data, how="outer").join(humid_data, how="outer").join(forecast_data, how="outer")
    joined = joined.sort_index()
    
    # Compute hourly features
    joined["temp_diff_int_ext"] = joined["temp_int"] - joined["temp_ext"]
    joined["humid_diff_int_ext"] = joined["humid_int"] - joined["humid_ext"]
    joined = joined[joined.index.notna()]
    joined = joined.dropna(subset=['temp_ext', 'temp_int'])

    return joined

def build_daily_features(joined_data, config_file):
    """Build daily aggregated features with forecast data included."""
    # Daily aggregations for internal and external actual data
    daily = joined_data.resample("1D").agg({
        "temp_ext": ["mean", "min", "max"],
        "temp_int": ["mean", "min", "max"],
        "temp_forecast": ["mean", "min", "max"],
        "humid_ext": ["mean", "min", "max"],
        "humid_int": ["mean", "min", "max"],
        "humid_forecast": ["mean", "min", "max"],
        "sunshine_duration": ["sum", "mean", "max"],
        "sunshine_duration_forecast": ["sum", "mean", "max"],
        "temp_diff_int_ext": ["mean", "min", "max"],
        "humid_diff_int_ext": ["mean", "min", "max"],
    })

    # Flatten MultiIndex columns and add proper prefixes
    daily.columns = ["_".join(filter(None, (c if isinstance(c, tuple) else (c,)))) 
                    for c in daily.columns.to_flat_index()]
    
    # Rename columns with proper prefixes
    column_mapping = {
        # Internal data (Home Assistant)
        "temp_int_mean": "internal_temp_mean",
        "temp_int_min": "internal_temp_min", 
        "temp_int_max": "internal_temp_max",
        "humid_int_mean": "internal_humid_mean",
        "humid_int_min": "internal_humid_min",
        "humid_int_max": "internal_humid_max",
        
        # External actual data (OpenMeteo current)
        "temp_ext_mean": "external_temp_mean",
        "temp_ext_min": "external_temp_min",
        "temp_ext_max": "external_temp_max", 
        "humid_ext_mean": "external_humid_mean",
        "humid_ext_min": "external_humid_min",
        "humid_ext_max": "external_humid_max",
        "sunshine_duration_sum": "external_sunshine_sum",
        "sunshine_duration_mean": "external_sunshine_mean",
        "sunshine_duration_max": "external_sunshine_max",

        # External forecast data (OpenMeteo forecasting)
        "temp_forecast_mean": "forecast_temp_mean",
        "temp_forecast_min": "forecast_temp_min",
        "temp_forecast_max": "forecast_temp_max", 
        "humid_forecast_mean": "forecast_humid_mean",
        "humid_forecast_min": "forecast_humid_min",
        "humid_forecast_max": "forecast_humid_max",
        "sunshine_duration_forecast_sum": "forecast_sunshine_sum",
        "sunshine_duration_forecast_mean": "forecast_sunshine_mean",
        "sunshine_duration_forecast_max": "forecast_sunshine_max",

        # Computed fields
        "temp_diff_int_ext_mean": "computed_temp_diff_int_ext_mean",
        "temp_diff_int_ext_min": "computed_temp_diff_int_ext_min", 
        "temp_diff_int_ext_max": "computed_temp_diff_int_ext_max",
        "humid_diff_int_ext_mean": "computed_humid_diff_int_ext_mean",
        "humid_diff_int_ext_min": "computed_humid_diff_int_ext_min",
        "humid_diff_int_ext_max": "computed_humid_diff_int_ext_max",
    }
    
    daily = daily.rename(columns=column_mapping)
    
    # Compute deltas for internal and external data
    daily["internal_temp_delta"] = daily["internal_temp_max"] - daily["internal_temp_min"]
    daily["internal_humid_delta"] = daily["internal_humid_max"] - daily["internal_humid_min"]
    daily["external_temp_delta"] = daily["external_temp_max"] - daily["external_temp_min"]
    daily["external_humid_delta"] = daily["external_humid_max"] - daily["external_humid_min"]

    
    # Add freezing flag
    daily = _add_freezing_flag(daily)
    

    # # Compute moving averages for all features
    daily = compute_moving_averages(daily, config_file)

    # Filter out days without internal temperature data
    daily = daily[daily.get("internal_temp_mean").notnull()]
    
    # # Order columns
    # daily = _order_columns(daily)
    
    return daily

# def _order_columns(self, daily: pd.DataFrame) -> pd.DataFrame:
#     """Order columns with proper prefixes for the new structure."""
#     ordered_cols = [
#         # Internal data (Home Assistant)
#         "internal_temp_mean", "internal_temp_min", "internal_temp_max", "internal_temp_delta",
#         "internal_humid_mean", "internal_humid_min", "internal_humid_max", "internal_humid_delta",
        
#         # External actual data (OpenMeteo current)
#         "external_temp_mean", "external_temp_min", "external_temp_max", "external_temp_delta",
#         "external_humid_mean", "external_humid_min", "external_humid_max", "external_humid_delta",
#         "external_sunshine_sum", "external_sunshine_mean", "external_sunshine_max",
        
#         # Forecast data (OpenMeteo forecast)
#         "forecast_temp_mean", "forecast_temp_min", "forecast_temp_max", "forecast_temp_delta",
#         "forecast_humid_mean", "forecast_humid_min", "forecast_humid_max", "forecast_humid_delta",
#         "forecast_sunshine_sum", "forecast_sunshine_mean", "forecast_sunshine_max",
        
#         # Computed features
#         "computed_temp_diff_int_ext", "computed_humid_diff_int_ext",
        
#         # Moving averages for internal data (3-day)
#         "internal_temp_mean_MA3", "internal_temp_min_MA3", "internal_temp_max_MA3", "internal_temp_delta_MA3",
#         "internal_humid_mean_MA3", "internal_humid_min_MA3", "internal_humid_max_MA3", "internal_humid_delta_MA3",
        
#         # Moving averages for external data (3-day)
#         "external_temp_mean_MA3", "external_temp_min_MA3", "external_temp_max_MA3", "external_temp_delta_MA3",
#         "external_humid_mean_MA3", "external_humid_min_MA3", "external_humid_max_MA3", "external_humid_delta_MA3",
#         "external_sunshine_sum_MA3", "external_sunshine_mean_MA3", "external_sunshine_max_MA3",
        
#         # Moving averages for forecast data (3-day)
#         "forecast_temp_mean_MA3", "forecast_temp_min_MA3", "forecast_temp_max_MA3", "forecast_temp_delta_MA3",
#         "forecast_humid_mean_MA3", "forecast_humid_min_MA3", "forecast_humid_max_MA3", "forecast_humid_delta_MA3",
#         "forecast_sunshine_sum_MA3", "forecast_sunshine_mean_MA3", "forecast_sunshine_max_MA3",
        
#         # Moving averages for computed features (3-day)
#         "computed_temp_diff_int_ext_MA3", "computed_humid_diff_int_ext_MA3",
        
#         # Moving averages for internal data (5-day)
#         "internal_temp_mean_MA5", "internal_temp_min_MA5", "internal_temp_max_MA5", "internal_temp_delta_MA5",
#         "internal_humid_mean_MA5", "internal_humid_min_MA5", "internal_humid_max_MA5", "internal_humid_delta_MA5",
        
#         # Moving averages for external data (5-day)
#         "external_temp_mean_MA5", "external_temp_min_MA5", "external_temp_max_MA5", "external_temp_delta_MA5",
#         "external_humid_mean_MA5", "external_humid_min_MA5", "external_humid_max_MA5", "external_humid_delta_MA5",
#         "external_sunshine_sum_MA5", "external_sunshine_mean_MA5", "external_sunshine_max_MA5",
        
#         # Moving averages for forecast data (5-day)
#         "forecast_temp_mean_MA5", "forecast_temp_min_MA5", "forecast_temp_max_MA5", "forecast_temp_delta_MA5",
#         "forecast_humid_mean_MA5", "forecast_humid_min_MA5", "forecast_humid_max_MA5", "forecast_humid_delta_MA5",
#         "forecast_sunshine_sum_MA5", "forecast_sunshine_mean_MA5", "forecast_sunshine_max_MA5",
        
#         # Moving averages for computed features (5-day)
#         "computed_temp_diff_int_ext_MA5", "computed_humid_diff_int_ext_MA5",
        
#         # Moving averages for internal data (7-day)
#         "internal_temp_mean_MA7", "internal_temp_min_MA7", "internal_temp_max_MA7", "internal_temp_delta_MA7",
#         "internal_humid_mean_MA7", "internal_humid_min_MA7", "internal_humid_max_MA7", "internal_humid_delta_MA7",
        
#         # Moving averages for external data (7-day)
#         "external_temp_mean_MA7", "external_temp_min_MA7", "external_temp_max_MA7", "external_temp_delta_MA7",
#         "external_humid_mean_MA7", "external_humid_min_MA7", "external_humid_max_MA7", "external_humid_delta_MA7",
#         "external_sunshine_sum_MA7", "external_sunshine_mean_MA7", "external_sunshine_max_MA7",
        
#         # Moving averages for forecast data (7-day)
#         # "forecast_temp_mean_MA7", "forecast_temp_min_MA7", "forecast_temp_max_MA7", "forecast_temp_delta_MA7",
#         # "forecast_humid_mean_MA7", "forecast_humid_min_MA7", "forecast_humid_max_MA7", "forecast_humid_delta_MA7",
#         # "forecast_sunshine_sum_MA7", "forecast_sunshine_mean_MA7", "forecast_sunshine_max_MA7",
        
#         # Moving averages for computed features (7-day)
#         "computed_temp_diff_int_ext_MA7", "computed_humid_diff_int_ext_MA7",
        
#         # Target variables
#         "freezing_on_tomorrow"
#     ]
    
#     # Keep only existing columns
#     existing_cols = [c for c in ordered_cols if c in daily.columns]
#     return daily[existing_cols]

def compute_moving_averages(df, config_file):
    """Compute moving averages for all numeric columns."""
    
    windows = config_file['data_processing']['moving_averages']['windows']
    
    result_df = df.copy()
    
    # Get all numeric columns (exclude target variable)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    target_col = config_file['model']['target_column']
    if target_col in numeric_cols:
        numeric_cols = numeric_cols.drop(target_col)

    numeric_cols = [c for c in numeric_cols if "forecast" not in c]

    for col in numeric_cols:
        for window in windows:
            ma_col = f"{col}_MA{window}"
            result_df[ma_col] = df[col].rolling(window=window, min_periods=1).mean()
    
    return result_df

def _add_freezing_flag(daily):
    """Add freezing flag based on the original hardcoded rules."""
    idx = daily.index
    
    # Freezing flag per requested rules (from original build_features.py)
    start_to_apr01 = idx <= pd.Timestamp("2025-04-01", tz=idx.tz)
    oct26_to_end = idx >= pd.Timestamp("2025-10-26", tz=idx.tz)
    freezing = pd.Series(0, index=idx, dtype="int64")
    freezing[start_to_apr01] = 1
    freezing[oct26_to_end] = 1
    
    # NEW: Add tomorrow's freezing target (shifted by -1 day)
    daily["freezing_on_tomorrow"] = freezing
    
    return daily

def build_features(df_temp_int, df_humid_int, df_weather, df_forecast, config_file):
    temp_data = resample_to_hourly(df_temp_int, 'temp_int')
    humid_data = resample_to_hourly(df_humid_int, 'humid_int')
    weather_data = df_weather.copy()
    forecast_data = create_hourly_forecast(df_forecast, df_weather)
    joined = join_hourly_data(temp_data, humid_data, weather_data, forecast_data)
    daily = build_daily_features(joined, config_file)
    return daily