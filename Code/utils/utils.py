import yaml
from pathlib import Path
import requests
import json
import websocket

def _load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def setup_data_directory(base_path):
    """
    Create a standardized /Data directory structure:
    
    /Data
      ├── External_source
      ├── internal_source
      │     └── history
      └── Output
            └── model
    """
    data_dir = base_path / "Data"
    subdirs = [
        data_dir / "External_source",
        data_dir / "internal_source" / "history",
        data_dir / "Output" / "model",
    ]

    for subdir in subdirs:
        subdir.mkdir(parents=True, exist_ok=True)

    return data_dir

    


def check_first_run(sensors_dict: dict, history_dir: Path) -> bool:
    """
    Checks if all expected CSV files exist in the history directory.
    If at least one is missing:
      - Deletes all existing CSVs in history_dir
      - Returns True (first run)
    Otherwise returns False.
    """
    csv_files = [history_dir / f"{name}_history.csv" for name in sensors_dict.keys()]

    # Check if all exist
    all_exist = all(f.exists() for f in csv_files)

    if not all_exist:
        # Cleanup any existing CSVs
        for f in history_dir.glob("*.csv"):
                f.unlink()
        return True

    return False

def send_to_HA(config_file: dict, prediction: str, probability: float):
    ws_url = config_file['data_sources']['home_assistant']['ws_url']
    token = config_file['data_sources']['home_assistant']['token']
    pred_freez_id = config_file['data_sources']['home_assistant']['sensors']['prediction']['id']
    pred_freez_prob_id = config_file['data_sources']['home_assistant']['sensors']['prediction_prob']['id']
    ws = websocket.create_connection(ws_url)

    # Étape 0 : lire le message auth_required
    auth_req = json.loads(ws.recv())

    # Étape 1 : envoyer le token
    auth_message = {"type": "auth", "access_token": token}
    ws.send(json.dumps(auth_message))
    auth_resp = json.loads(ws.recv())

    if auth_resp.get("type") != "auth_ok":
        return

    # Étape 2 : call_service
    service = "turn_on" if prediction else "turn_off"
    message_pred = {
        "id": 1,
        "type": "call_service",
        "domain": "input_boolean",
        "service": service,
        "target": {"entity_id": pred_freez_id}
        # "target": {"entity_id": "input_boolean.light_off"}
    }

    message_prob = {
        "id": 2,
        "type": "call_service",
        "domain": "input_number",
        "service": "set_value",
        "target": {"entity_id": pred_freez_prob_id},
        "service_data": {"value": round(probability*100,2)}
    }


    ws.send(json.dumps(message_pred))
    ws.send(json.dumps(message_prob))

    ws.close()
