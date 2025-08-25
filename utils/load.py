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


def load_json_from_files(filepath, num_lines=None):
    """
    Create DataFrame from the .json files avec possibilité de limiter le nombre de lignes.

    Args:
        filepath (str): Chemin vers le fichier JSON
        num_lines (int, optional): Nombre maximum de lignes à charger. Si None, charge tout.

    Returns:
        pd.DataFrame: DataFrame contenant les données
    """
    records = []
    line_count = 0

    try:
        with jsonlines.open(filepath, mode='r') as reader:
            for obj in reader:
                records.append(obj)
                line_count += 1

                # Arrêter si on a atteint le nombre de lignes demandé
                if num_lines is not None and line_count >= num_lines:
                    break
    except Exception as generalError:
        print(f"Error reading file {filepath}: {generalError}")
        return pd.DataFrame()

    print(f"Loaded {len(records)} lines from {filepath}")
    return pd.DataFrame(records)
