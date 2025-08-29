import numpy as np
import pandas as pd

# from pathlib import Path
# from colorama import Fore, Style
# from dateutil.parser import parse

from medium.params import *
from medium.ml_logic.data import clean_data, load_json_from_files,create_dataframe_to_predict
from medium.ml_logic.registry import load_model, save_model, save_results, save_preprocessor, load_preprocessor

from medium.ml_logic.model import initialize_model, compile_model, train_model, evaluate_model, implemented_model
from medium.ml_logic.preprocessor import preprocess_features,preprocess_pred
from sklearn.metrics import mean_absolute_error
import time

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
    df_processed, preprocessor = preprocess_features(data_cleaned)

    # Sauvegarder les données traitées localement si necessaire
    df_processed.to_csv(os.path.join(PATH_DATA, f"df_processed_{DATA_SIZE}.csv"), index=False)

    # Sauvegarder le préprocesseur
    save_preprocessor(preprocessor)

    print("✅main preprocess done \n")

    return None


def train(model_name:str, split_ratio: float = 0.2 ):
    """
    - Charge les données préprocessées
    - Entraîne le modèle sur le dataset
    - Stocke les résultats et les poids du modèle

    Return val_mae as a float
    """
    print("🎬 main train starting ................\n")

    # Charger les données préprocessées (depuis le csv si sauvegardé)
    df_processed = pd.read_csv(os.path.join(PATH_DATA, f"df_processed_{DATA_SIZE}.csv"))

    # Créer X et y
    X = df_processed.drop(columns=['log1p_recommends'])
    y = df_processed['log1p_recommends']

    # Split train/validation
    train_length = int(len(df_processed)*(1-split_ratio))
    X_train, X_val = X[:train_length], X[train_length:]
    y_train, y_val = y[:train_length], y[train_length:]

    model = load_model(model_name)

    if model is None:
        # Initialiser le modèle
        model = initialize_model(model_name = model_name)
        # entrainement aucun model trouvé
        model = train_model(model=model, X=X_train, y=y_train)

    val_metric = evaluate_model(model=model, X=X_val, y=y_val)

    params = {
        "split_ratio": split_ratio,
        "metric": implemented_model[model.__class__.__name__]['metrics']
    }

    # Save results
    save_results(model_name,params=params, metrics=dict(mae=val_metric))

    # Save model
    save_model(model=model)

    print("✅ main train() done \n")
    return val_metric


def evaluate(model_name:str, X_pred: pd.DataFrame = None, ):
    """
    Évalue la performance du modèle sur l'ensemble de validation
    Return metric as a float
    """
    print("🎬 main evaluate starting ................\n")

    if X_pred is None:
        # chqarger les test pour évaluer
        X_pred = load_json_from_files(X_filepath=DATA_TEST, y_filepath=DATA_TEST_LOG_RECOMMEND, num_lines=DATA_TEST_SIZE)
        old_pred = X_pred['log1p_recommends'].copy()
        X_pred.drop(columns=['log1p_recommends'])

    data_cleaned = clean_data(X_pred)

    model = load_model(model_name)
    preprocessor = load_preprocessor()

    print(f" ℹ️ the model type : {model.__class__.__name__} ... ")

    X_processed = preprocess_pred(data_cleaned,preprocessor)
    print(f" ℹ️ X_processed shape :  { X_processed.shape}")

    y_pred = model.predict(X_processed)

    # Transformation inverse si nécessaire
    y_pred_original = np.expm1(y_pred)

    mae = mean_absolute_error(old_pred , y_pred)


     # Supposons que X_pred a une colonne 'prediction' avec les anciennes valeurs
    results_df = pd.DataFrame({
    'old_pred': old_pred,
    'new_pred': y_pred,
    'nb_reco': y_pred_original
     })
    # Sauvegarder les predictions
    date_run = time.strftime("%Y%m%d-%H%M%S")
    target_path= os.path.join(PATH_METRICS, f"metrics_{DATA_SIZE}_{model.__class__.__name__}_{date_run}.csv")
    results_df.to_csv(target_path, index=False)
    print(f" ✅ = = = = = = = = => mean_absolute_error : {mae} \n")
    print(f" ✅ evaluate done \n")
    return mae



def pred(model_name:str, text: str="",title:str=""):
    """
    Fait une prédiction using le dernier modèle entraîné
    """
    print("🎬 pred starting ................\n")

    X_pred = create_dataframe_to_predict(text,title)
    data_cleaned = clean_data(X_pred)

    model = load_model(model_name)
    preprocessor = load_preprocessor()
    print(f" ℹ️ the model type : {model.__class__.__name__} ... ")
    print(f" ℹ️ the text : {text} ... ")
    X_processed = preprocess_pred(data_cleaned,preprocessor)
    y_pred = model.predict(X_processed)
    print(f" ℹ️ the probabilty (log1p) : {y_pred} ... ")

    nb_recommandation = np.expm1(y_pred)
    print(f" ℹ️ the nb of claps : {nb_recommandation} ... ")

    # Sauvegarder les predictions
    # date_run = time.strftime("%Y%m%d-%H%M%S")
    # target_path= os.path.join(PATH_PREDICTION, f"predictions_{DATA_SIZE}_{model.__class__.__name__}_{date_run}.csv")
    # results_df.to_csv(target_path, index=False)

    print(f" ✅ pred() end \n")
    return nb_recommandation


def run_all(model_name:str):
    print("🎬 run all starting ................\n")
    preprocess()
    train(model_name)
    evaluate(model_name)
    # pred(model_name)
    print(f" ✅ run all end.\n")

if __name__ == '__main__':
    # Workflow complet
    # preprocess()
    # train()
    # evaluate()
    # pred()
    pass
