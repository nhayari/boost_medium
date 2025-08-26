import pandas as pd

def convert_dict_columns(df: pd.DataFrame, drop_cols = True):
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
