import pandas as pd
import jsonlines
import pickle
import os

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


def load_csv_old(filepath, num_lines=None):
    """
    Lit le fichier CSV et crée un DataFrame.

    Args:
        filepath (str): Chemin du fichier CSV
        num_lines (int, optional): Nombre de lignes à lire

    Returns:
        pd.DataFrame: DataFrame avec les colonnes '_id' et 'log1p_recommends'
    """
    if num_lines:
        df = pd.read_csv(filepath, nrows=num_lines)
    else:
        df = pd.read_csv(filepath)

    # Renommer les colonnes pour être cohérent
    df = df.rename(columns={'id': '_id', 'log_recommends': 'log1p_recommends'})

    return df


def load_json(filepath, nlines=None, startline=0):
    """
    Créer un DataFrame  à partir du .json avec possibilité de limiter le nombre de lignes
    et de commencer à partir d'une ligne spécifique.

    Args:
        filepath (str): Chemin vers le fichier JSON
        nlines (int, optional): Nombre maximum de lignes à charger. Si None, charge tout.
        startline (int, optional): Ligne à partir de laquelle commencer la lecture (0-indexed)

    Returns:
        pd.DataFrame: DataFrame contenant les données
    """
    records = []
    line_count = 0
    current_line = 0

    try:
        with jsonlines.open(filepath, mode='r') as reader:
            for obj in reader:
                # Ignorer les lignes avant startline
                if current_line < startline:
                    current_line += 1
                    continue

                records.append(obj)
                line_count += 1
                current_line += 1

                # Arrêter si on a atteint le nombre de lignes demandé
                if nlines is not None and line_count >= nlines:
                    break

    except Exception as generalError:
        print(f"Error reading file {filepath}: {generalError}")
        return pd.DataFrame()

    print(f"Loaded {len(records)} lines from {filepath} (starting from line {startline})")
    return pd.DataFrame(records)


def load_csv(filepath, nlines=None, startline=0):
    """
    Lit le fichier CSV et crée un DataFrame avec possibilité de limiter le nombre de lignes
    et de commencer à partir d'une ligne spécifique.

    Args:
        filepath (str): Chemin du fichier CSV
        nlines (int, optional): Nombre de lignes à lire. Si None, lit tout le fichier.
        startline (int, optional): Ligne à partir de laquelle commencer la lecture (0-indexed)

    Returns:
        pd.DataFrame: DataFrame avec les colonnes '_id' et 'log1p_recommends'
    """
    # Calculer le nombre de lignes à sauter (header + startline)
    skiprows = startline + 1 if startline > 0 else None

    # Déterminer le nombre de lignes à lire
    nrows = nlines

    # Lire le fichier avec les paramètres appropriés
    df = pd.read_csv(
        filepath,
        skiprows=skiprows,
        nrows=nrows
    )

    # Si on a sauté des lignes, réinitialiser l'index
    if startline > 0:
        df = df.reset_index(drop=True)

    # Renommer les colonnes pour être cohérent
    df = df.rename(columns={'id': '_id', 'log_recommends': 'log1p_recommends'})

    print(f"Loaded {len(df)} lines from {filepath} (starting from line {startline})")
    return df




def load_pickle(filepath):
    """
    Charge un fichier pickle

    Args:
        filepath (str): Chemin vers le fichier .pkl

    Returns:
        Objet chargé depuis le fichier pickle ou None en cas d'erreur
    """
    try:
        # Vérifier si le fichier existe
        if not os.path.exists(filepath):
            print(f"🛑 Le fichier {filepath} n'existe pas")
            return None

        # Charger le fichier
        with open(filepath, 'rb') as file:
            data = pickle.load(file)

        print(f"✅ Fichier {filepath} chargé avec succès")
        return data

    except Exception as e:
        print(f"🛑Erreur lors du chargement du fichier pickle {filepath}: {e}")
        return None
