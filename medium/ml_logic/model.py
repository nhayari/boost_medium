import numpy as np
import time
from scipy import stats
from sklearn.metrics import mean_absolute_error

# from typing import Tuple
# from tensorflow import keras
# from keras import Sequential, layers, regularizers, optimizers
# from keras.callbacks import EarlyStopping

from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor


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
