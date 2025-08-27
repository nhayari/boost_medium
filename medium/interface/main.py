import numpy as np
import pandas as pd

# from pathlib import Path
# from colorama import Fore, Style
# from dateutil.parser import parse

from medium.params import *
from medium.ml_logic.data import clean_data, load_json_from_files
from medium.ml_logic.registry import load_model, save_model, save_results

from medium.ml_logic.model import initialize_model, compile_model, train_model, evaluate_model, implemented_model
from medium.ml_logic.preprocessor import preprocess_features

def preprocess() -> None:
    """
    - Charge les données brutes depuis les fichiers JSON et CSV
    - Nettoie et préprocesse les données
    - Stocke les données traitées
    """
    print("🎬 main preprocess starting ................\n")

    # Charger les données JSON
    data = load_json_from_files(X_filepath=DATA_TRAIN, y_filepath=DATA_LOG_RECOMMEND, num_lines=DATA_SIZE)

    # Nettoyer les données
    data_cleaned = clean_data(data)

    # Prétraiter les features
    df_processed = preprocess_features(data_cleaned)

    # Sauvegarder les données traitées localement si necessaire
    df_processed.to_csv(os.path.join(LOCAL_REGISTRY_PATH, "data", f"df_processed_{DATA_SIZE}.csv"), index=False)

    print("🏁 main preprocess done \n")

    return None

def train(
        split_ratio: float = 0.2,
        #batch_size=32,
        #patience=3
    ) -> float:
    """
    - Charge les données préprocessées
    - Entraîne le modèle sur le dataset
    - Stocke les résultats et les poids du modèle

    Return val_mae as a float
    """
    print("🎬 main train starting ................\n")

    # Charger les données préprocessées (despuis le csv si sauvegardé)
    df_processed = pd.read_csv(os.path.join(LOCAL_REGISTRY_PATH, "data", f"df_processed_{DATA_SIZE}.csv"))

    # Créer X et y
    X = df_processed.drop(columns=['log1p_recommends'])
    y = df_processed['log1p_recommends']

    # Split train/validation
    train_length = int(len(df_processed)*(1-split_ratio))
    X_train, X_val = X[:train_length], X[train_length:]
    y_train, y_val = y[:train_length], y[train_length:]

    model = load_model()

    if model is None:
        # Initialiser le modèle
        model = initialize_model(model = 'LinearRegression', input_shape=(X_train.shape[1],))

    model = train_model(model=model, X=X_train, y=y_train)

    val_metric = evaluate_model(model=model, X=X_val, y=y_val)

    params = {
        "split_ratio": split_ratio,
        "metric": implemented_model[model.__class__.__name__]['metrics']
    }

    # Save results
    save_results(params=params, metrics=dict(mae=val_metric))

    # Save model
    save_model(model=model)

    print("🏁 main train() done \n")
    return val_metric


def evaluate(stage: str = "Production") -> float:
    """
    Évalue la performance du modèle sur l'ensemble de validation
    Return metric as a float
    """
    print("🎬 main evaluate starting ................\n")

    metric =0.0

    print(" 💤 TO DO   !!!!!!!!!!!!!!  \n")


    print("🏁 main evaluate() done \n")
    return metric


def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
    """
    Fait une prédiction using le dernier modèle entraîné
    """
    print("🎬 pred starting ................\n")

    metric =0.0

    print(" 💤 TO DO   !!!!!!!!!!!!!! \n")

    print("\n⭐️ Use case: predict")

    if X_pred is None:
        # Exemple de données pour prédiction
        # 🫡 à prévoir  le webstrapping via URL en focntion de l'avancement
        X_pred = pd.DataFrame({
            'content': ['This is a sample article about machine learning and data science.'],
            'title': ['Machine learning'],
            'author': ['Data Scientist'],
            'published': ['2025-08-26T14:06:00Z']
        })

    model = None  #load_model()

    X_processed =None
    # y_pred = model.predict(X_processed)

    # Transformation inverse si nécessaire
    # !! expm1   = exp -1
    y_pred_original = None
    # y_pred_original = np.expm1(y_pred) #  log1p inverse

    print("🏁 pred() done \n")
    return y_pred_original

def run_all():
    preprocess()
    train()
    #evaluate()
    #pred()

if __name__ == '__main__':
    # Workflow complet
    preprocess()
    train()
    evaluate()
    pred()
