import requests
import pandas as pd
from datetime import timedelta

def fetch_weather(start_date, end_date, config_file, type_fetch):
    """Fetch historical weather data."""
    params_request = {
        "latitude": config_file['data_sources']['openmeteo']['coordinates']['latitude'],
        "longitude": config_file['data_sources']['openmeteo']['coordinates']['longitude'],
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "sunshine_duration"
        ],
        "timezone": config_file['data_sources']['openmeteo']['coordinates']['timezone']
        }
    archive_url = config_file['data_sources']['openmeteo']['archive_url']
    base_url = config_file['data_sources']['openmeteo']['base_url']

    if type_fetch == 'historical':
        request_url =  archive_url + "/v1/archive"
    elif type_fetch == 'forecast':
        request_url = base_url + "/v1/forecast"

    response = requests.get(request_url, params=params_request, timeout=30)
    response.raise_for_status()

    data = response.json()

    # Convert to DataFrame
    df = pd.DataFrame({
        "datetime": data["hourly"]["time"],
        "temp_ext": data["hourly"]["temperature_2m"],
        "humid_ext": data["hourly"]["relative_humidity_2m"],
        "sunshine_duration": [s/3600 for s in data["hourly"]["sunshine_duration"]]  # seconds → hours
    })

    # Convert datetime to timezone-aware
    df["datetime"] = pd.to_datetime(df["datetime"], utc=False)
    df["datetime"] = df["datetime"].dt.tz_localize(params_request["timezone"], 
                                                   nonexistent="shift_forward", 
                                                   ambiguous="NaT"
                                                   )

    return df