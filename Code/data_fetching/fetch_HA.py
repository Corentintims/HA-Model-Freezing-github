from datetime import datetime
import pandas as pd
import requests

def ha_date_formator(dt, type_boundaries):
    """
    Convert a date in 'YYYYMMDD' format to Home Assistant-compatible
    ISO 8601 start and end timestamps (UTC style).
    """
    if type_boundaries == 'start':
        date_str_formated = dt.strftime("%Y-%m-%dT00:00:00Z")
    elif type_boundaries == 'end':
        date_str_formated = dt.strftime("%Y-%m-%dT23:59:59Z")
    return date_str_formated

def fetch_sensor_history(start_date, end_date, config_file, sensor_info):
    """Fetch historical data for a specific sensor with chunking for large date ranges."""

    entity_id = sensor_info['id']
    data_col = sensor_info['data_col']
    
    # Parse dates
    start_time = ha_date_formator(start_date, type_boundaries='start')
    end_time = ha_date_formator(end_date, type_boundaries='end')

    headers = {
            "Authorization": f"Bearer {config_file['data_sources']['home_assistant']['token']}",
            "Content-Type": "application/json"
        }
    url = f"{config_file['data_sources']['home_assistant']['base_url']}/api/history/period/{start_time}"
    params = {
        "filter_entity_id": entity_id,
        "end_time": end_time
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=60)
        response.raise_for_status()
        
        data = response.json()
        
        if not data or not data[0]:
            print(f"No data found for {entity_id} in chunk {start_time} to {end_time}")
            return
        
        # Convert to DataFrame
        records = []
        for entry in data[0]:
            records.append({
                "entity_id": entry["entity_id"],
                "state": entry["state"],
                "last_changed": entry["last_changed"],
                "last_updated": entry["last_updated"]
            })

        df = pd.DataFrame(records)    
         # Process the data
        if data_col in ['temp_int', 'humid_int']:
            df[data_col] = pd.to_numeric(df["state"], errors="coerce")
        else:
            df[data_col] = df["state"].map({'on': 1, 'off': 0}).astype('Int64')
        df["last_changed"] = pd.to_datetime(df["last_changed"], utc=True, format='ISO8601')
        df["datetime"] = df["last_changed"].dt.tz_convert("Europe/Brussels")
        
        result = df[["datetime", data_col]].copy()
        result = result.dropna(subset=[data_col])
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {entity_id}: {e}")
        return
    except Exception as e:
        print(f"Unexpected error for {entity_id}: {e}")
        return