from medium.params import *
from medium.ml_logic.data import load_json_from_files

# from medium.ml_logic.data import clean_data, load_json_from_files,create_dataframe_to_predict
from medium.ml_logic.registry import load_model,load_preprocessor
# , save_model, save_results, save_preprocessor,

from medium.ml_logic.model import predict_pipeline
from medium.ml_logic.preprocessor import preprocess_features
# from medium.ml_logic.model import (
#     initialize_model, compile_model, train_model, evaluate_model, implemented_model,
#     create_medium_pipeline, train_pipeline, predict_pipeline, evaluate_pipeline
# )
# from medium.ml_logic.preprocessor import (
#     preprocess_features, preprocess_pred, MediumPreprocessingPipeline
# )
# from sklearn.metrics import mean_absolute_error




from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
import time


def run_grid_search(estimator, param_grid, X_train, y_train, X_test=None, y_test=None,
                   cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1,
                   return_train_score=False, refit=True):
    """
    Exécute un GridSearchCV complet avec évaluation et reporting détaillé

    Args:
        estimator: modèle sklearn (RandomForestRegressor, etc.)
        param_grid: dictionnaire de paramètres à tester
        X_train, y_train: données d'entraînement
        X_test, y_test: données de test (optionnelles pour évaluation)
        cv: nombre de folds pour cross-validation
        scoring: métrique d'évaluation
        n_jobs: nombre de jobs parallèles (-1 = tous les cores)
        verbose: niveau de verbosité
        return_train_score: retourner les scores d'entraînement
        refit: ré-entraîner sur tout le dataset après grid search

    Returns:
        grid_search: objet GridSearchCV entraîné
        results_df: DataFrame avec les résultats détaillés
    """

    print(f"🚀 Starting GridSearchCV for {estimator.__class__.__name__}")
    print(f"📊 Parameter grid: {len(param_grid)} combinations")
    print(f"🎯 Scoring: {scoring}")
    print(f"📈 CV folds: {cv}")

    # Démarrage du timer
    start_time = time.time()

    # Configuration du GridSearch
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=return_train_score,
        refit=refit
    )

    # Entraînement
    grid_search.fit(X_train, y_train)

    # Fin du timer
    end_time = time.time()
    duration = end_time - start_time

    print(f"✅ GridSearch completed in {duration:.2f} seconds ({duration/60:.2f} minutes)")

    # Affichage des résultats
    print("\n" + "="*10)
    print(" GRID SEARCH RESULTS")
    print("="*10)

    print(f" Best parameters: {grid_search.best_params_}")
    print(f" Best CV score ({scoring}): {grid_search.best_score_:.4f}")

    if scoring == 'neg_mean_absolute_error':
        print(f" Best MAE: {-grid_search.best_score_:.4f}")
    elif scoring == 'neg_mean_squared_error':
        print(f" Best MSE: {-grid_search.best_score_:.4f}")
        print(f" Best RMSE: {np.sqrt(-grid_search.best_score_):.4f}")

    # Évaluation sur le test set si fourni
    if X_test is not None and y_test is not None:
        print("\n TEST SET EVALUATION")
        print("-" * 30)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"MAE:  {mae:.4f}")

    # Création du DataFrame des résultats
    results_df = pd.DataFrame(grid_search.cv_results_)

    # Tri par score
    results_df = results_df.sort_values('mean_test_score', ascending=False)

    return grid_search, results_df




def save_best_model_params(grid_search, model_name="_",filename="best_model"):
    """
       # Sauvegarde du best model en CSV
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    if grid_search.best_estimator_ is not None:
        params_df = pd.DataFrame([grid_search.best_params_])
        params_df['best_score'] = grid_search.best_score_
        params_df['best_mae'] = -grid_search.best_score_
        params_df.to_csv(f"{filename}_{model_name}_{DATA_SIZE}_{timestamp}.csv", index=False)
        print(f" Best model saved in {filename}_{model_name}_{DATA_SIZE}_{timestamp}.csv")



def create_sample(model_name: str, X_filepath):
    """
    create sample
    """
    print("🎬 create_sample................\n")

    if X_filepath is None:
        return

    pipeline = load_model(f"{model_name}_pipeline")

    if pipeline is None :
        return

    # Charger les données de test
    df_test = load_json_from_files(X_filepath=DATA_TEST)

    try:
        # Charger le pipeline complet
        pipeline = load_model(f"{model_name}_pipeline")

        if pipeline is not None and hasattr(pipeline, 'named_steps'):
            print(f"ℹ️ Using complete pipeline: {type(pipeline)}")

            # Faire des prédictions pour les résultats détaillés
            y_pred = predict_pipeline(pipeline, df_test, errors='ignore')

            # Assurer l'alignement des données (le pipeline peut avoir filtré des lignes)
            preprocessed_data = pipeline.named_steps['preprocessor'].transform(df_test)
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
            y_pred = model.predict(X_processed)
        else:
            raise ValueError("Model not found")

    # Transformation inverse si nécessaire
    # y_pred_original = np.expm1(y_pred)

    # Créer DataFrame des résultats

    results_df = pd.DataFrame({
        'id': df_test['id'],
        'log_recommends': y_pred
      })

    # Sauvegarder les prédictions
    date_run = time.strftime("%Y%m%d-%H%M%S")
    model_type = "Unknown"
    if 'pipeline' in locals() and pipeline is not None and hasattr(pipeline, 'named_steps'):
        model_type = type(pipeline.named_steps['model']).__name__
    elif 'model' in locals() and model is not None:
        model_type = model.__class__.__name__

    target_path = os.path.join(PATH_METRICS, f"sample_submission.csv_{model_type}_{date_run}.csv")
    results_df.to_csv(target_path, index=False)
    print(f"✅ create_sample")
