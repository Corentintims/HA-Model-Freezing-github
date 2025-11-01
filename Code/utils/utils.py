import yaml
from pathlib import Path

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
            try:
                f.unlink()
            except Exception as e:
                print(f"⚠️ Could not delete {f}: {e}")
        return True

    return False
