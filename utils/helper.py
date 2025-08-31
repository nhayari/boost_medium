from medium.params import *
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
