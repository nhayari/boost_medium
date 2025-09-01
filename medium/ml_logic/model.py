import numpy as np
import pandas as pd
import time
from scipy import stats
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin
from typing import Optional

# from typing import Tuple
# from tensorflow import keras
# from keras import Sequential, layers, regularizers, optimizers
# from keras.callbacks import EarlyStopping

from sklearn.linear_model import LinearRegression, ElasticNet,Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
# from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from medium.ml_logic.preprocessor import MediumPreprocessingPipeline


implemented_model = {
    'LinearRegression': {
        'metrics': ['mae']
       },
    'RandomForestRegressor': {
        'metrics': ['mae'],
        'n_estimators':50
    },
    'ExtraTreesRegressor': {
        'metrics': ['mae'],
        'n_estimators':100,
        'bootstrap': False,
        'max_depth': None,
        'max_features': 'sqrt',
        'min_samples_leaf': 1,
        'min_samples_split': 5
    },
    'ElasticNet' : {
        'metrics' : ['mae'],
        'L1_ratio':stats.uniform(0, 1),
        'alpha':stats.uniform(0, 10)
    },
    'Ridge' : {
        'metrics' : ['mae'],
        'alpha':1.0
    },
    'XGBRegressor' : {
        'metrics' : ['mae'],
       'n_estimators': 200,
       'learning_rate': 0.05,
       'eval_metric': 'mae',
       'objective' :'reg:squarederror'
    },
    'LGBMRegressor': {
        'metrics' : ['mae'],
        'n_estimators': 200,
        'learning_rate':0.05,
        'metric': 'mae',
        'objective' : 'regression'
    },
    'GradientBoostingRegressor': {
        'metrics' : ['mae'],
        'n_estimators': 100,
        'learning_rate':0.1,
        'loss':'absolute_error'
    }
}


def initialize_model(model_name = 'LinearRegression'):
    """
    Initialize the Model

    Args:
        model_name (str): Name of the model to initialize

    Returns:
        Model: Initialized model instance
    """
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
    """
    Compile if necessary
    """
    #   print("✅ Model initialized")
    print("🎬 compile_model starting ................\n")
    # optimizer = optimizers.Adam(learning_rate=learning_rate)
    # model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=["mae"])
    print("✅ Model compiled")
    return model


def train_model(model, X=None, y=None):
    """
    Fit the model and return model
    """
    print("🎬 train_model starting ................\n")

    if X is None or y is None or len(X) == 0 or len(y) == 0 or len(X) != len(y):
        print("⚠️ Skipping model training due to invalid data ! None or Length")
        return model

    # Effectuer l'entraînement
    model.fit(X, y)
    print("✅ train_model() done \n")
    return model


def evaluate_model (model, X=None, y=None):
    """
    Evaluate trained model performance on the dataset
    """
    metrics = {}

    print("🎬 evaluate_model starting ................\n")

    for metric in implemented_model[model.__class__.__name__]['metrics']:
        if metric == 'mae':
            y_pred = model.predict(X)
            if y is not None:
                metrics[metric] = mean_absolute_error(y, y_pred)
        else:
            print("⚠️ Skipping  !! metric must be added in implemented_model.")
            print("🏁 evaluate_model() end \n")

    print("✅ evaluate_model() done \n")

    return metrics






# model_dict= implemented_model[model_name]
def getExtraTreesRegressor(model_dict):
    """
    Initialise model ExtraTreeRegressor with params
    Args:
        model_dict (dict): params

    Returns:
        Model: Initialized model instance
    """
    model = ExtraTreesRegressor(n_estimators=model_dict['n_estimators'], bootstrap=model_dict['bootstrap'],
                                max_depth=model_dict['max_depth'],max_features=model_dict['max_features'], min_samples_leaf=model_dict['min_samples_leaf'],
                                min_samples_split=model_dict['min_samples_split'])

    return model


# def getXGBRegressor(model_dict):
#     """
#     Initialize XGBRegressor with parameters
#     Args:
#         model_dict (dict): parameters dictionary

#     Returns:
#         XGBRegressor: Initialized XGBoost model instance
#     """
#     if not XGBOOST_AVAILABLE:
#         raise ImportError("XGBoost is not installed. Install with: pip install xgboost")

#     model = xgb.XGBRegressor(
#         n_estimators=model_dict['n_estimators'],
#         max_depth=model_dict['max_depth'],
#         learning_rate=model_dict['learning_rate'],
#         subsample=model_dict['subsample'],
#         colsample_bytree=model_dict['colsample_bytree'],
#         random_state=model_dict['random_state'],
#         n_jobs=model_dict['n_jobs'],
#         objective=model_dict['objective'],
#         eval_metric=model_dict['eval_metric']
#     )

#     return model


def create_medium_pipeline(
    model_name: str = 'LinearRegression',
    datetime_col: str = 'published_$date',
    text_columns: Optional[list] = None,
    html_columns: Optional[list] = None,
    chunk_size: int = 1000,
    remove_punct: bool = True,
    remove_stopwords: bool = True,
    tf_idf_min_ratio: float = 0.02,
    show_progress: bool = True
) -> Pipeline:
    """
    Create a complete pipeline with preprocessing and modeling.

    Args:
        model_name (str): Name of the model to use
        Other args: Parameters for preprocessing pipeline

    Returns:
        Pipeline: Complete sklearn pipeline with preprocessing and model
    """
    print("🎬 Creating Medium pipeline...")

    # Create preprocessing pipeline
    preprocessor = MediumPreprocessingPipeline(
        datetime_col=datetime_col,
        text_columns=text_columns,
        html_columns=html_columns,
        chunk_size=chunk_size,
        remove_punct=remove_punct,
        remove_stopwords=remove_stopwords,
        tf_idf_min_ratio=tf_idf_min_ratio,
        show_progress=show_progress
    )

    # Initialize model
    model = initialize_model(model_name)

    # Create complete pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    print("✅ Medium pipeline created successfully!")
    return pipeline


def train_pipeline(pipeline: Pipeline, X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    Train the complete pipeline.

    Args:
        pipeline (Pipeline): The pipeline to train
        X (pd.DataFrame): Input data
        y (pd.Series, optional): Target variable. If None, assumes it's in X['log1p_recommends']

    Returns:
        Pipeline: Trained pipeline
    """
    print("🎬 Training pipeline...")

    # If target variable is in the dataframe, let the preprocessor handle it
    # The preprocessor will keep target and features aligned during cleaning
    if y is None and 'log1p_recommends' in X.columns:
        # The pipeline will handle the full dataframe including target variable
        # The preprocessor is designed to keep target and features aligned

        # First, get preprocessed data with target variable
        preprocessed_data = pipeline.named_steps['preprocessor'].fit_transform(X)

        # Extract aligned target and features
        if 'log1p_recommends' in preprocessed_data.columns:
            y_aligned = preprocessed_data['log1p_recommends']
            X_aligned = preprocessed_data.drop(columns=['log1p_recommends'])
        else:
            raise ValueError("Target variable 'log1p_recommends' not found in preprocessed data")

        # Train the model with aligned data
        pipeline.named_steps['model'].fit(X_aligned, y_aligned)

    else:
        # If target is provided separately, use traditional approach
        X_train = X.drop(columns=['log1p_recommends'], errors='ignore')
        if y is None:
            raise ValueError("Target variable 'y' must be provided or 'log1p_recommends' column must exist in X")
        pipeline.fit(X_train, y)

    print("✅ Pipeline training completed!")
    return pipeline


def predict_pipeline(pipeline: Pipeline, X: pd.DataFrame):
    """
    Make predictions using the trained pipeline.

    Args:
        pipeline (Pipeline): Trained pipeline
        X (pd.DataFrame): Input data

    Returns:
        np.ndarray: Predictions
    """
    print("🎬 Making predictions with pipeline...")

    # Remove target variable if it exists
    X_pred = X.drop(columns=['log1p_recommends'], errors='ignore')

    # Make predictions
    predictions = pipeline.predict(X_pred)

    print("✅ Predictions completed!")
    return predictions


def evaluate_pipeline(pipeline: Pipeline, X: pd.DataFrame, y: Optional[pd.Series] = None) -> dict:
    """
    Evaluate the trained pipeline.

    Args:
        pipeline (Pipeline): Trained pipeline
        X (pd.DataFrame): Input data
        y (pd.Series, optional): True target values. If None, assumes it's in X['log1p_recommends']

    Returns:
        dict: Evaluation metrics
    """
    print("🎬 Evaluating pipeline...")

    # Handle alignment issue similar to training
    if y is None and 'log1p_recommends' in X.columns:
        # Preprocess the data to get aligned target and features
        preprocessed_data = pipeline.named_steps['preprocessor'].transform(X)

        # Extract aligned target and features
        if 'log1p_recommends' in preprocessed_data.columns:
            y_true = preprocessed_data['log1p_recommends']
            X_eval = preprocessed_data.drop(columns=['log1p_recommends'])
        else:
            raise ValueError("Target variable 'log1p_recommends' not found in preprocessed data")

        # Make predictions with the model directly (data already preprocessed)
        y_pred = pipeline.named_steps['model'].predict(X_eval)
    else:
        # If target is provided separately, use traditional approach
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

    print("✅ Pipeline evaluation completed!")
    return metrics
def getRidge(model_dict):
    """
    Initialise model Ridge with params
    Args:
        model_dict (dict): params
    Returns:
        Model: Initialized model instance
    """
    model = Ridge(alpha=model_dict['alpha'])
    return model

def getXGBRegressor(model_dict):
    """
    Initialise model XGBRegressor with params
    Args:
        model_dict (dict): params

    Returns:
        Model: Initialized model instance
    """
    model = XGBRegressor(n_estimators=model_dict['n_estimators'],
                         learning_rate=model_dict['learning_rate'],
                         eval_metric=model_dict['eval_metric'],
                         objective=model_dict['objective']
                         )
    return model

def getGradientBoostingRegressor(model_dict):
    """
    Initialise model GradientBoostingRegressor with params
    Args:
        model_dict (dict): params

    Returns:
        Model: Initialized model instance
    """
    model = GradientBoostingRegressor(n_estimators=model_dict['n_estimators'],
                         learning_rate=model_dict['learning_rate'],
                         loss=model_dict['loss']
                         )
    return model


def getLGBMRegressor(model_dict):
    """
    Initialise model LGBMRegressor with params
    Args:
        model_dict (dict): params

    Returns:
        Model: Initialized model instance
    """
    model = LGBMRegressor(n_estimators=model_dict['n_estimators'],
                         learning_rate=model_dict['learning_rate'],
                         metric=model_dict['metric'],
                         objective=model_dict['objective']
                         )
    return model
