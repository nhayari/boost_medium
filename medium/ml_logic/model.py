import numpy as np
import time

from sklearn.metrics import mean_absolute_error

# from typing import Tuple
# from tensorflow import keras
# from keras import Sequential, layers, regularizers, optimizers
# from keras.callbacks import EarlyStopping

from sklearn.linear_model import LinearRegression

implemented_model = {
    'LinearRegression': {
        'metrics': ['mae']
    }
}

def initialize_model(model = 'LinearRegression', input_shape: tuple = None):
    #  -> Model:
    """
    Initialize the Neural Network with random weights
    """

    if model not in implemented_model:
        raise ValueError(f"Model '{model}' is not implemented.")

    print("🎬 initialize_model starting ................\n")

    if model == 'LinearRegression':
        model = LinearRegression()

    print("🏁 initialize_model() done \n")

    return model


def compile_model(model, learning_rate=0.0005):
    """
    Compile if necessary
    """
    #   print("✅ Model initialized")
    print("🎬 compile_model starting ................\n")
    modele = None
    print("🏁 compile_model() done \n")
    print("✅ Model compiled")
    return model

def train_model(model, X=None, y=None):
    """
    Fit the model and return  model or tuple (fitted_model, history)
    """
    print("🎬 train_model starting ................\n")
    metrics = None
    model.fit(X, y)
    print("🏁 train_model() done \n")

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

    print("🏁 evaluate_model() done \n")

    return metrics
