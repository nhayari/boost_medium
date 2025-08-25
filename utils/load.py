import pandas as pd
import jsonlines

def load_json_files(filepath):
    """Create DataFrame from the .json files."""
    records = []

    try:
        with jsonlines.open(filepath, mode='r') as reader:
            for obj in reader:
                records.append(obj)
    except jsonlines.InvalidLineError as e:
        print(f"Invalid line encountered: {e}")
        pass
    return pd.DataFrame(records)
