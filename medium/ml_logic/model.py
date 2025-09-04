import numpy as np
import pandas as pd
import time
from scipy import stats
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin
from typing import Optional, Tuple
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from medium.ml_logic.preprocessor import MediumPreprocessingPipeline


implemented_model = {
    'LinearRegression': {
        'metrics': ['mae']
    },
    'RandomForestRegressor': {
        'metrics': ['mae'],
        'n_estimators': 50
    },
    'ExtraTreesRegressor': {
        'metrics': ['mae'],
        'n_estimators': 100,
        'bootstrap': False,
        'max_depth': None,
        'max_features': 'sqrt',
        'min_samples_leaf': 1,
        'min_samples_split': 5
    },
    'ElasticNet': {
        'metrics': ['mae'],
        'L1_ratio': stats.uniform(0, 1),
        'alpha': stats.uniform(0, 10)
    },
    'Ridge': {
        'metrics': ['mae'],
        'alpha': 1.0
    },
    'XGBRegressor': {
        'n_estimators': 500,
        'learning_rate': 0.05,
        'max_depth': 6,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0,
        'reg_lambda': 1,
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'mae',
        'metrics': ['mae'],
        'objective': 'reg:absoluteerror'
    },
    'LGBMRegressor': {
        'metrics': ['mae'],
        'n_estimators': 200,
        'learning_rate': 0.05,
        'metric': 'mae',
        'objective': 'regression'
    },
    'GradientBoostingRegressor': {
        'metrics': ['mae'],
        'n_estimators': 100,
        'learning_rate': 0.1,
        'loss': 'absolute_error'
    }
}


def initialize_model(model_name='LinearRegression'):
    """Initialize the Model"""
    if model_name not in implemented_model:
        implemented_models_list = list(implemented_model.keys())
        raise ValueError(
            f"Model '{model_name}' is not implemented.\n"
            f"Implemented models are: {implemented_models_list}"
        )

    print("🎬 initialize_model starting ................\n")

    if model_name == 'LinearRegression':
        print(f"ℹ️ Model: LinearRegression \n")
        model = LinearRegression()
    elif model_name == 'RandomForestRegressor':
        print(f"ℹ️ Model: RandomForestRegressor, n_estimators: {implemented_model[model_name]['n_estimators']} \n")
        model = RandomForestRegressor(n_estimators=implemented_model[model_name]['n_estimators'])
    elif model_name == 'ExtraTreesRegressor':
        print(f"ℹ️ Model: ExtraTreesRegressor, n_estimators: {implemented_model[model_name]['n_estimators']} \n")
        model = getExtraTreesRegressor(implemented_model[model_name])
    elif model_name == 'ElasticNet':
        print(f"ℹ️ Model: ElasticNet \n")
        model = ElasticNet()
    elif model_name == 'Ridge':
        print(f"ℹ️ Model: Ridge \n")
        model = getRidge(implemented_model[model_name])
    elif model_name == 'XGBRegressor':
        print(f"ℹ️ Model: XGBRegressor \n")
        model = getXGBRegressor(implemented_model[model_name])
    elif model_name == 'LGBMRegressor':
        print(f"ℹ️ Model: LGBMRegressor \n")
        model = getLGBMRegressor(implemented_model[model_name])
    elif model_name == 'GradientBoostingRegressor':
        print(f"ℹ️ Model: GradientBoostingRegressor \n")
        model = getGradientBoostingRegressor(implemented_model[model_name])

    print("✅ initialize_model() done \n")

    return model


def compile_model(model, learning_rate=0.0005):
    """Compile if necessary (for future neural network support)"""
    print("🎬 compile_model starting ................\n")
    # For sklearn models, no compilation needed
    print("✅ Model compiled (sklearn model - no compilation needed)")
    return model


def train_model(model, X=None, y=None):
    """Fit the model and return model"""
    print("🎬 train_model starting ................\n")

    if X is None or y is None or len(X) == 0 or len(y) == 0 or len(X) != len(y):
        print("⚠️ Skipping model training due to invalid data! None or Length")
        return model

    # Train the model
    model.fit(X, y)
    print("✅ train_model() done \n")
    return model


def evaluate_model(model, X=None, y=None):
    """Evaluate trained model performance on the dataset"""
    metrics = {}

    print("🎬 evaluate_model starting ................\n")

    for metric in implemented_model[model.__class__.__name__]['metrics']:
        if metric == 'mae':
            y_pred = model.predict(X)
            if y is not None:
                metrics[metric] = mean_absolute_error(y, y_pred)
        else:
            print("⚠️ Skipping! metric must be added in implemented_model.")
            print("🏁 evaluate_model() end \n")

    print("✅ evaluate_model() done \n")

    return metrics


def split_and_create_pipeline(
    data: pd.DataFrame,
    model_name: str = 'LinearRegression',
    split_ratio: float = 0.2,
    datetime_col: str = 'published_$date',
    text_columns: Optional[list] = None,
    html_columns: Optional[list] = None,
    chunk_size: int = 1000,
    remove_punct: bool = False,
    remove_stopwords: bool = False,
    tf_idf_min_ratio: float = 0.02,
    content_only: bool = False,
    metadata_only: bool = False,
    model_is_tree: bool = False,
    show_progress: bool = True
) -> Tuple[Pipeline, pd.DataFrame, pd.DataFrame]:
    """
    Split data FIRST, then create and fit pipeline to prevent data leakage.

    Returns:
        Tuple of (fitted_pipeline, train_data, val_data)
    """
    print("🎬 Creating leak-free pipeline with proper data splitting...")

    # CRITICAL: Split BEFORE any preprocessing
    train_length = int(len(data) * (1 - split_ratio))
    data_train = data[:train_length].copy()
    data_val = data[train_length:].copy()

    print(f"📊 Split data: {len(data_train)} train, {len(data_val)} validation")

    if content_only and metadata_only:
        print('Not possible to do both metadata-only and content-only.')
        print('Defaulting to content only.')
        metadata_only = False

    tree_models = ['RandomForestRegressor', 'ExtraTreesRegressor', 'GradientBoostingRegressor', 'XGBRegressor']

    if model_name in tree_models:
        model_is_tree = True

    # Create preprocessing pipeline
    preprocessor = MediumPreprocessingPipeline(
        datetime_col=datetime_col,
        text_columns=text_columns,
        html_columns=html_columns,
        chunk_size=chunk_size,
        remove_punct=remove_punct,
        remove_stopwords=remove_stopwords,
        tf_idf_min_ratio=tf_idf_min_ratio,
        content_only=content_only,
        metadata_only=metadata_only,
        model_is_tree=model_is_tree,
        show_progress=show_progress
    )

    # Initialize model
    model = initialize_model(model_name)

    # Create complete pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # FIT ONLY ON TRAINING DATA
    print("🔧 Fitting pipeline on training data only...")

    # Separate target from features for training data
    if 'log1p_recommends' in data_train.columns:
        y_train = data_train['log1p_recommends']
        X_train = data_train.drop(columns=['log1p_recommends'])
        pipeline.fit(X_train, y_train)
    else:
        raise ValueError("Target variable 'log1p_recommends' not found in training data")

    print("✅ Pipeline created and fitted on training data only!")
    return pipeline, data_train, data_val


def train_pipeline(pipeline: Pipeline, X: pd.DataFrame, y: Optional[pd.Series] = None,
                   is_preprocessed: bool = False):
    """
    Train the pipeline with proper handling of preprocessed data.

    FIXED: Ensures no data leakage when training pipeline.
    """
    print("🎬 Training pipeline...")

    if is_preprocessed:
        # Data is already preprocessed, just train the model part
        if y is None and 'log1p_recommends' in X.columns:
            y = X['log1p_recommends']
            X = X.drop(columns=['log1p_recommends'])

        if y is None:
            raise ValueError("Target variable must be provided for preprocessed data")

        # Train only the model component
        pipeline.named_steps['model'].fit(X, y)

    else:
        # Data needs preprocessing
        if y is None and 'log1p_recommends' in X.columns:
            # Pipeline will handle the full dataframe
            y = X['log1p_recommends']
            X = X.drop(columns=['log1p_recommends'])

        if y is None:
            raise ValueError("Target variable 'y' must be provided or 'log1p_recommends' column must exist in X")

        # Fit the entire pipeline
        pipeline.fit(X, y)

    print("✅ Pipeline training completed!")
    return pipeline


def predict_pipeline(pipeline: Pipeline, X: pd.DataFrame):
    """Make predictions using the trained pipeline."""
    print("🎬 Making predictions with pipeline...")

    # Remove target variable if it exists
    X_pred = X.drop(columns=['log1p_recommends'], errors='ignore')

    # Make predictions
    predictions = pipeline.predict(X_pred)

    print("✅ Predictions completed!")
    return predictions


def evaluate_pipeline(pipeline: Pipeline, X: pd.DataFrame, y: Optional[pd.Series] = None,
                      is_preprocessed: bool = False) -> dict:
    """
    Evaluate the trained pipeline.

    FIXED: Proper handling of preprocessed vs raw data.
    """
    print("🎬 Evaluating pipeline...")

    if is_preprocessed:
        # Data is already preprocessed
        if y is None and 'log1p_recommends' in X.columns:
            y_true = X['log1p_recommends']
            X_eval = X.drop(columns=['log1p_recommends'])
        else:
            X_eval = X.drop(columns=['log1p_recommends'], errors='ignore')
            y_true = y

        if y_true is None:
            raise ValueError("Target variable must be provided for evaluation")

        # Make predictions with model directly (data already preprocessed)
        y_pred = pipeline.named_steps['model'].predict(X_eval)

    else:
        # Data needs preprocessing
        if y is None and 'log1p_recommends' in X.columns:
            y_true = X['log1p_recommends']
            X_eval = X.drop(columns=['log1p_recommends'])
        else:
            X_eval = X.drop(columns=['log1p_recommends'], errors='ignore')
            y_true = y

        if y_true is None:
            raise ValueError("Target variable 'y' must be provided or 'log1p_recommends' column must exist in X")

        # Make predictions using full pipeline
        y_pred = pipeline.predict(X_eval)

    # Calculate metrics
    metrics = {}
    model_name = pipeline.named_steps['model'].__class__.__name__

    if model_name in implemented_model:
        for metric in implemented_model[model_name]['metrics']:
            if metric == 'mae':
                metrics[metric] = mean_absolute_error(y_true, y_pred)

    print(f"✅ Pipeline evaluation completed: MAE = {metrics.get('mae', 'N/A')}")
    return metrics


# Helper functions for specific models
def getExtraTreesRegressor(model_dict):
    """Initialize ExtraTreesRegressor with params"""
    model = ExtraTreesRegressor(
        n_estimators=model_dict['n_estimators'],
        bootstrap=model_dict['bootstrap'],
        max_depth=model_dict['max_depth'],
        max_features=model_dict['max_features'],
        min_samples_leaf=model_dict['min_samples_leaf'],
        min_samples_split=model_dict['min_samples_split']
    )
    return model


def getRidge(model_dict):
    """Initialize Ridge with params"""
    model = Ridge(alpha=model_dict['alpha'])
    return model


def getXGBRegressor(model_dict):
    """Initialize XGBRegressor with params"""
    model = XGBRegressor(
        n_estimators=model_dict['n_estimators'],
        learning_rate=model_dict['learning_rate'],
        eval_metric=model_dict['eval_metric'],
        objective=model_dict['objective'],
        n_jobs=model_dict['n_jobs'],
        subsample=model_dict['subsample'],
        colsample_bytree=model_dict['colsample_bytree']
    )
    return model


def getGradientBoostingRegressor(model_dict):
    """Initialize GradientBoostingRegressor with params"""
    model = GradientBoostingRegressor(
        n_estimators=model_dict['n_estimators'],
        learning_rate=model_dict['learning_rate'],
        loss=model_dict['loss']
    )
    return model


def getLGBMRegressor(model_dict):
    """Initialize LGBMRegressor with params"""
    model = LGBMRegressor(
        n_estimators=model_dict['n_estimators'],
        learning_rate=model_dict['learning_rate'],
        metric=model_dict['metric'],
        objective=model_dict['objective']
    )
    return model
