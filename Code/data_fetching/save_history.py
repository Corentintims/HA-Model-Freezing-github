import pandas as pd

def save_sensor_history(df, csv_path, first_run):
    """
    Save or update the CSV containing the historical values of each HA sensor.
    - If first_run: create or overwrite CSV entirely
    - Else: append only the last day of data if it’s not already in the CSV
    """
    # Ensure datetime column is parsed properly
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    if first_run or not csv_path.exists():
        df.to_csv(csv_path, index=False)
        return

    # Load existing CSV
    existing = pd.read_csv(csv_path)


    # Find the last date in the new DataFrame
    last_date = df["datetime"].dt.date.max()

    # Filter to only that last day
    df_last_day = df[df["datetime"].dt.date == last_date]

    # Determine which dates are already present in the CSV
    existing_dates = df["datetime"].dt.date.unique()
    print(existing_dates)

    # Append only if that date is not already in the CSV
    if last_date not in existing_dates:
        df_last_day.to_csv(csv_path, mode="a", index=False, header=False)
