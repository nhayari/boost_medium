import numpy as np
import pandas as pd

# from pathlib import Path
# from colorama import Fore, Style
# from dateutil.parser import parse

from medium.params import *
from medium.ml_logic.data import clean_data, load_json_from_files,create_dataframe_to_predict
from medium.ml_logic.registry import load_model, save_model, save_results, save_preprocessor, load_preprocessor

from medium.ml_logic.model import (
    initialize_model, compile_model, train_model, evaluate_model, implemented_model,
    create_medium_pipeline, train_pipeline, predict_pipeline, evaluate_pipeline
)
from medium.ml_logic.preprocessor import (
    preprocess_features, preprocess_pred, MediumPreprocessingPipeline
)
from sklearn.metrics import mean_absolute_error
import time
import pickle


def save_model_with_custom_name(model, custom_name: str) -> bool:
    """
    Save model with a custom name instead of using model.__class__.__name__
    """
    try:
        print(f"🎬 save_model_with_custom_name starting ({custom_name})................\n")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_path = os.path.join(PATH_MODELS, f"{custom_name}_{DATA_SIZE}_{timestamp}.pickle")
        with open(model_path, "wb") as file:
            pickle.dump(model, file)
        print(f" ✅ save_model_with_custom_name() done \n")
        return True
    except Exception as e:
        print(f"Error saving model with custom name: {e}")
        return False

def preprocess() -> None:
    """
    - Charge les données brutes depuis les fichiers JSON et CSV
    - Nettoie et préprocesse les données avec le nouveau pipeline transformer
    - Stocke les données traitées et le preprocesseur
    """
    print("🎬 main preprocess starting (transformer approach)................\n")

    # Charger les données JSON
    data = load_json_from_files(X_filepath=DATA_TRAIN, y_filepath=DATA_LOG_RECOMMEND, num_lines=DATA_SIZE)

    # Créer et entraîner le pipeline de préprocessing
    preprocessor = MediumPreprocessingPipeline(
        chunk_size=200,
        remove_punct=True,
        remove_stopwords=True,
        show_progress=True
    )
    
    # Fit et transform les données
    df_processed = preprocessor.fit_transform(data)

    # Sauvegarder les données traitées localement (s'assurer que c'est un DataFrame)
    if hasattr(df_processed, 'to_csv'):
        df_processed.to_csv(os.path.join(PATH_DATA, f"df_processed_{DATA_SIZE}.csv"), index=False) # type: ignore
    else:
        # Convertir en DataFrame si nécessaire
        pd.DataFrame(df_processed).to_csv(os.path.join(PATH_DATA, f"df_processed_{DATA_SIZE}.csv"), index=False)

    # Sauvegarder le preprocesseur complet (nouveau pipeline)
    save_preprocessor(preprocessor, "medium_pipeline_preprocessor")
    
    # Sauvegarder aussi les composants individuels pour compatibilité
    save_preprocessor(preprocessor.tfidf_vectorizer, "tfidf_vectorizer")
    save_preprocessor(preprocessor.std_scaler, "standard_scaler")

    print("✅main preprocess done (transformer approach)\n")

    return None


def train(model_name: str, split_ratio: float = 0.2):
    """
    - Charge les données préprocessées
    - Entraîne le modèle avec le nouveau pipeline transformer
    - Stocke les résultats et les poids du modèle

    Return val_mae as a float
    """
    print("🎬 main train starting (transformer approach)................\n")

    # Option 1: Utiliser le pipeline complet (recommandé)
    try:
        # Charger les données brutes pour le pipeline complet
        data = load_json_from_files(X_filepath=DATA_TRAIN, y_filepath=DATA_LOG_RECOMMEND, num_lines=DATA_SIZE)
        
        # Créer le pipeline complet (preprocessing + model)
        pipeline = create_medium_pipeline(
            model_name=model_name,
            chunk_size=200,
            remove_punct=True,
            remove_stopwords=True,
            show_progress=True
        )
        
        # Split train/validation sur les données brutes
        train_length = int(len(data) * (1 - split_ratio))
        data_train, data_val = data[:train_length], data[train_length:]
        
        # Entraîner le pipeline complet
        trained_pipeline = train_pipeline(pipeline, data_train)
        
        # Évaluer le pipeline
        val_metrics = evaluate_pipeline(trained_pipeline, data_val)
        val_mae = val_metrics.get('mae', 0.0)
        
        # Sauvegarder le pipeline complet
        save_model_with_custom_name(trained_pipeline, f"{model_name}_pipeline")
        
    except Exception as e:
        print(f"❌ Pipeline approach failed: {e}")
        print("🔄 Falling back to traditional approach...")
        
        # Option 2: Fallback vers l'approche traditionnelle
        df_processed = pd.read_csv(os.path.join(PATH_DATA, f"df_processed_{DATA_SIZE}.csv"))
        
        # Créer X et y
        X = df_processed.drop(columns=['log1p_recommends'])
        y = df_processed['log1p_recommends']
        
        # Split train/validation
        train_length = int(len(df_processed) * (1 - split_ratio))
        X_train, X_val = X[:train_length], X[train_length:]
        y_train, y_val = y[:train_length], y[train_length:]
        
        # Initialiser et entraîner le modèle
        model = initialize_model(model_name=model_name)
        model = train_model(model=model, X=X_train, y=y_train)
        
        # Évaluer
        val_metric = evaluate_model(model=model, X=X_val, y=y_val)
        val_mae = val_metric.get('mae', 0.0) if isinstance(val_metric, dict) else val_metric
        
        # Sauvegarder le modèle traditionnel
        save_model(model=model)

    params = {
        "split_ratio": split_ratio,
        "approach": "transformer_pipeline",
        "model_name": model_name
    }

    # Save results
    save_results(model_name, params=params, metrics=dict(mae=val_mae))

    print("✅ main train() done (transformer approach)\n")
    return val_mae


def evaluate(model_name: str, df_test: pd.DataFrame | None = None):
    """
    Évalue la performance du modèle avec le nouveau pipeline transformer
    Return metric as a float
    """
    print("🎬 main evaluate starting (transformer approach)................\n")

    if df_test is None:
        # Charger les données de test
        df_test = load_json_from_files(X_filepath=DATA_TEST, y_filepath=DATA_TEST_LOG_RECOMMEND, num_lines=DATA_TEST_SIZE)

    # Option 1: Essayer d'utiliser le pipeline complet (recommandé)
    try:
        # Charger le pipeline complet
        pipeline = load_model(f"{model_name}_pipeline")
        
        if pipeline is not None and hasattr(pipeline, 'named_steps'):
            print(f"ℹ️ Using complete pipeline: {type(pipeline)}")
            
            # Évaluer avec le pipeline
            metrics = evaluate_pipeline(pipeline, df_test)
            mae = metrics.get('mae', 0.0)
            
            # Faire des prédictions pour les résultats détaillés
            y_pred = predict_pipeline(pipeline, df_test.drop(columns=['log1p_recommends'], errors='ignore'))
            old_pred = df_test['log1p_recommends'].values
            
            # Assurer l'alignement des données (le pipeline peut avoir filtré des lignes)
            preprocessed_data = pipeline.named_steps['preprocessor'].transform(df_test)
            if 'log1p_recommends' in preprocessed_data.columns:
                old_pred = preprocessed_data['log1p_recommends'].values
        else:
            raise ValueError("Pipeline not found or invalid")
        
    except Exception as e:
        print(f"❌ Pipeline approach failed: {e}")
        print("🔄 Falling back to traditional approach...")
        
        # Option 2: Fallback vers l'approche traditionnelle
        model = load_model(model_name)
        tfidf_preprocessor = load_preprocessor('tfidf_vectorizer')
        std_preprocessor = load_preprocessor('standard_scaler')
        
        if model is not None:
            print(f"ℹ️ the model type: {model.__class__.__name__}")
            
            X_processed, __tfidf, __scaler = preprocess_features(
                df_test, 
                chunksize=200, 
                remove_punct=True, 
                remove_stopwords=True, 
                tfidf_vectorizer=tfidf_preprocessor, 
                std_scaler=std_preprocessor
            )
            
            print(f"ℹ️ X_processed shape: {X_processed.shape}")
            old_pred = X_processed['log1p_recommends'].copy()
            y_pred = model.predict(X_processed.drop(columns=['log1p_recommends']))
            
            mae = mean_absolute_error(old_pred, y_pred)
        else:
            raise ValueError("Model not found")

    # Transformation inverse si nécessaire
    y_pred_original = np.expm1(y_pred)

    # Créer DataFrame des résultats
    min_length = min(len(old_pred), len(y_pred), len(y_pred_original))
    results_df = pd.DataFrame({
        'old_pred': old_pred[:min_length],
        'new_pred': y_pred[:min_length],
        'nb_reco': y_pred_original[:min_length]
    })
    
    # Sauvegarder les prédictions
    date_run = time.strftime("%Y%m%d-%H%M%S")
    model_type = "Unknown"
    if 'pipeline' in locals() and pipeline is not None and hasattr(pipeline, 'named_steps'):
        model_type = type(pipeline.named_steps['model']).__name__
    elif 'model' in locals() and model is not None:
        model_type = model.__class__.__name__
    
    target_path = os.path.join(PATH_METRICS, f"metrics_{DATA_SIZE}_{model_type}_{date_run}.csv")
    results_df.to_csv(target_path, index=False)
    
    print(f"✅ = = = = = = = = => mean_absolute_error: {mae}")
    print(f"✅ evaluate done (transformer approach)")
    return mae



def pred(model_name: str, text: str = "", title: str = ""):
    """
    Fait une prédiction using le nouveau pipeline transformer
    """
    print("🎬 pred starting (transformer approach)................\n")

    # Créer le DataFrame de prédiction
    X_pred = create_dataframe_to_predict(text, title)
    print(f"ℹ️ the text: {text}")
    print(f"ℹ️ the title: {title}")

    # Option 1: Essayer d'utiliser le pipeline complet (recommandé)
    try:
        # Charger le pipeline complet
        pipeline = load_model(f"{model_name}_pipeline")
        
        if pipeline is not None and hasattr(pipeline, 'named_steps'):
            print(f"ℹ️ Using complete pipeline: {type(pipeline)}")
            
            # Faire la prédiction avec le pipeline
            y_pred = predict_pipeline(pipeline, X_pred)
            
            model_type = type(pipeline.named_steps['model']).__name__
            print(f"ℹ️ the model type: {model_type}")
        else:
            raise ValueError("Pipeline not found or invalid")
        
    except Exception as e:
        print(f"❌ Pipeline approach failed: {e}")
        print("🔄 Falling back to traditional approach...")
        
        # Option 2: Fallback vers l'approche traditionnelle
        data_cleaned = clean_data(X_pred)
        
        model = load_model(model_name)
        preprocessor = load_preprocessor('tfidf_vectorizer')  # Utiliser le TF-IDF vectorizer
        
        if model is not None:
            print(f"ℹ️ the model type: {model.__class__.__name__}")
            
            X_processed = preprocess_pred(data_cleaned, preprocessor)
            y_pred = model.predict(X_processed)
        else:
            raise ValueError("Model not found")

    print(f"ℹ️ the probability (log1p): {y_pred}")
    
    # Transformation inverse
    nb_recommandation = np.expm1(y_pred)
    print(f"ℹ️ the nb of claps: {nb_recommandation}")

    # Optionnel: Sauvegarder les prédictions
    # date_run = time.strftime("%Y%m%d-%H%M%S")
    # target_path = os.path.join(PATH_PREDICTION, f"predictions_{model_type}_{date_run}.csv")
    # results_df.to_csv(target_path, index=False)

    print(f"✅ pred() end (transformer approach)")
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
