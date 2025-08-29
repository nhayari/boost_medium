import pandas as pd
import jsonlines
import re
from html import unescape

from medium.params import *

HTML_TAG_PATTERN = re.compile(r'<[^>]+>')
WHITESPACE_PATTERN = re.compile(r'\s+')

def load_json_from_files(X_filepath, y_filepath, num_lines: int | None = None) -> pd.DataFrame:
    """
    Create DataFrame from the .json files avec possibilité de limiter le nombre de lignes.

    Args:
        X_filepath (str): Chemin vers le fichier JSON
        num_lines (int, optional): Nombre maximum de lignes à charger. Si None, charge tout.

    Returns:
        pd.DataFrame: DataFrame contenant les données
    """
    records = []
    line_count = 0

    try:
        with jsonlines.open(X_filepath, mode='r') as reader:
            for obj in reader:
                records.append(obj)
                line_count += 1
                # Arrêter si on a atteint le nombre de lignes demandé
                if num_lines is not None and line_count >= int(num_lines):
                    break
    except Exception as generalError:
        print(f"Error reading file {X_filepath}: {generalError}")
        return pd.DataFrame()

    print(f"Loaded {len(records)} lines from {X_filepath}")

    df = pd.DataFrame(records)
    df_y = load_csv(y_filepath, num_lines=num_lines)


    df_final = pd.concat([df, df_y['log1p_recommends']], axis=1)

    print(f"Final DataFrame shape after concatenation: {df_final.shape}")

    return df_final

def load_csv(filepath, num_lines: int | None = None) -> pd.DataFrame:
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



def convert_dict_columns(df: pd.DataFrame, drop_cols: bool = True) -> pd.DataFrame:
    """
    Extrait et aplatit les colonnes contenant des dictionnaires dans un DataFrame pandas.

    Cette fonction transforme chaque clé de dictionnaire en une nouvelle colonne distincte,
    permettant de normaliser des données JSON ou des structures imbriquées.

    Parameters
    ----------
    df : pandas.DataFrame
        Le DataFrame contenant les colonnes à traiter
    drop_cols : bool, default True
        Si True, supprime les colonnes originales après extraction

    Returns
    -------
    pandas.DataFrame
        DataFrame modifié avec les nouvelles colonnes extraites

    """
    # Créer une copie pour éviter de modifier l'original
    df_result = df.copy()

    # Colonnes à supprimer à la fin
    cols_to_drop = []

    for col in df.columns:
        # Vérifier si toutes les valeurs sont des dictionnaires
        if df[col].apply(lambda c: isinstance(c, dict)).all():

            # Utiliser json_normalize pour aplatir la colonne
            dict_list = df[col].tolist()
            normalized = pd.json_normalize(dict_list)

            # Renommer les colonnes avec le préfixe de la colonne originale
            normalized.columns = [f'{col}_{col_name}' for col_name in normalized.columns]

            # Conserver l'index original pour l'alignement
            normalized.index = df.index

            # Ajouter les nouvelles colonnes au DataFrame
            df_result = pd.concat([df_result, normalized], axis=1)

            # Marquer la colonne pour suppression si demandé
            if drop_cols:
                cols_to_drop.append(col)

    # Supprimer les colonnes originales si demandé
    if cols_to_drop:
        df_result.drop(columns=cols_to_drop, inplace=True)

    return df_result

def title_contains_high_non_ascii_ratio(title: str, threshold: float = 0.2) -> bool:
    """
    Check if the title contains a high ratio of non-ASCII characters.

    Parameters:
    -----------
    title : str
        The title string to check.
    threshold : float
        The ratio threshold above which the title is considered to have high non-ASCII content.

    Returns:
    --------
    bool
        True if the ratio of non-ASCII characters exceeds the threshold, False otherwise.
    """
    if not isinstance(title, str) or len(title) == 0:
        return False

    non_ascii_count = sum(1 for char in title if ord(char) > 127)
    ratio = non_ascii_count / len(title)

    return ratio > threshold

def remove_constant_columns(df: pd.DataFrame, exclude_cols: list | str | None = None) -> pd.DataFrame:
    """
    Remove columns with only one unique value.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    exclude_cols : list, str, or None
        Column name(s) to exclude from constant checking.
        These columns will be kept regardless of their variance.
        Can be a single column name (str) or list of column names.

    Returns:
    --------
    pandas.DataFrame
        Dataframe with constant columns removed (except excluded ones)

    Examples:
    ---------
    # Exclude single column
    df_cleaned = remove_constant_columns_optimized(df, exclude_cols='id')

    # Exclude multiple columns
    df_cleaned = remove_constant_columns_optimized(df, exclude_cols=['id', 'timestamp', 'user_id'])
    """
    # Handle exclude_cols parameter
    if exclude_cols is None:
        exclude_cols = []
    elif isinstance(exclude_cols, str):
        exclude_cols = [exclude_cols]
    elif not isinstance(exclude_cols, (list, tuple, set)):
        raise ValueError("exclude_cols must be None, string, or list-like")

    # Convert to set for faster lookup
    exclude_set = set(exclude_cols)

    # Validate that excluded columns exist in dataframe
    missing_cols = exclude_set - set(df.columns)
    if missing_cols:
        raise ValueError(f"Excluded columns not found in dataframe: {missing_cols}")

    cols_to_keep = []
    for col in df.columns:
        # Keep column if it's excluded OR has more than 1 unique value
        if col in exclude_set:
            cols_to_keep.append(col)
        elif df[col].nunique() > 1:
            cols_to_keep.append(col)

    return df[cols_to_keep]

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data by flattening dict columns, removing constant columns,
    and stripping HTML tags from 'content' column.
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe to clean
    chunk_size : int
        Number of rows to process at a time when stripping HTML tags
    Returns:
    --------
    pandas.DataFrame
        Cleaned dataframe
    Notes:
    ------
    - Assumes 'content' column exists for HTML stripping
    - Excludes 'content' and 'tags' columns from constant removal
    - Prints progress messages during cleaning steps
    """

    print("🎬 Data cleaning started...\n")

    if not df.empty:
        print(f"Initial data shape: {df.shape}")

        # 1. Flatten dictionary columns
        df = convert_dict_columns(df, drop_cols=True)
        print(f" - Flattened dictionary columns, total columns now: {df.shape[1]}")

        # 2. Remove constant columns
        df = remove_constant_columns(df, exclude_cols=['content', 'tags'])
        print(f" - Removed constant columns, remaining columns: {df.shape[1]}")

        # 3. Remove articles where their title contains 20% of non-ASCII characters
        if 'title' in df.columns:
            initial_count = len(df)
            df = df[~df['title'].apply(title_contains_high_non_ascii_ratio)]
            removed_count = initial_count - len(df)
            print(f" - Removed {removed_count} articles with high non-ASCII ratio in title")
        else:
            print(" - 'title' column not found; skipping non-ASCII title filtering")

    print("✅ Data cleaned")
    return df


def create_dataframe_to_predict(text:str="", title:str=""):
    df = pd.DataFrame({
            'content': [text],
            'title': [title],
            'tags':[""]
    }
    )
    return df
