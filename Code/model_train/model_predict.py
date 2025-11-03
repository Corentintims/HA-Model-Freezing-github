from datetime import date, timedelta, datetime
import sys
from pathlib import Path
import joblib
import json

# Add project root to path (go up two levels from main folder to get parent of Code folder)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from Code.utils.features_building import build_features
from Code.data_fetching.fetch_HA import fetch_sensor_history
from Code.data_fetching.fetch_openmeto import fetch_weather
from Code.data_fetching.save_history import save_sensor_history
from Code.utils.utils import _load_config, setup_data_directory, check_first_run

def load_model(model_dir):
    model_path = model_dir / "best_model.pkl"
    metadata_path = model_dir / "metadata.json"

    # Charger le modèle
    model = joblib.load(model_path)

    # Charger les métadonnées si elles existent
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return model, metadata



def make_predictions():
    """Make predictions for a specific date."""
    # print("Making predictions...")
    """Main function for testing data utilities."""
    setup_data_directory(project_root)
    config_path = "config.yaml"
    config_file = _load_config(config_path)
    model_dir = Path(config_file['paths']['model_dir'])


    sensors_dict = config_file['data_sources']['home_assistant']['sensors']
    history_dir = Path(config_file['paths']['history_dir'])
    history_dir.mkdir(parents=True, exist_ok=True)
    # Get target date
    target_date = date.today()
    end_date = target_date - timedelta(days=1)
    start_date = end_date - timedelta(days=7)

    first_run = check_first_run(sensors_dict, history_dir)




    df_sensors = []
    for sensor_name, sensor_info in sensors_dict.items():
        csv_path = history_dir / f"{sensor_name}_history.csv"
        df_sensor = fetch_sensor_history(start_date, end_date, config_file, sensor_info=sensor_info)
        df_sensors.append(df_sensor)
        save_sensor_history(df_sensor, csv_path, first_run)

    df_weather  = fetch_weather(start_date, end_date, config_file, 'historical')
    df_forecast = fetch_weather(target_date, target_date, config_file, 'forecast')
    
    
    # Load historical data for feature generation        
    daily_features = build_features(df_sensors[0], df_sensors[1], df_weather, df_forecast, config_file)
    row_predict = daily_features[-1:]

    # Load model
    model, metadata = load_model(model_dir)
    
    # Make predictions using the full model
    predictions = model.predict(row_predict)[0]
    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(row_predict)[0]
    
    # # print results
    # print(f"\nPREDICTION RESULTS for {target_date}")
    # print("=" * 50)
    freezing_status = "FREEZING EXPECTED" if predictions == 1 else "No freezing expected"
    prob_text = f" (Probability: {probabilities[predictions]:.2f})"
    print(f"Date: {target_date} - {freezing_status}{prob_text}")
    
    # print("OK - Predictions completed successfully")

def main():
    make_predictions()

if __name__ == "__main__":
    main()
