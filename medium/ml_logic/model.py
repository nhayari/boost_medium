import numpy as np
import time

# from typing import Tuple
# from tensorflow import keras
# from keras import Sequential, layers, regularizers, optimizers
# from keras.callbacks import EarlyStopping

from sklearn.linear_model import LinearRegression

def initialize_model(input_shape: tuple):
    #  -> Model:
    """
    Initialize the Neural Network with random weights
    """
    model = LinearRegression()
    print("🎬 initialize_model starting ................\n")
    print(" 💤 TO DO   !!!!!!!!!!!!!! \n")
    print("🏁 initialize_model() done \n")
#   print("✅ Model initialized")
    return model


def compile_model(model, learning_rate=0.0005):
    """
    Compile if necessary
    """
    #   print("✅ Model initialized")
    print("🎬 compile_model starting ................\n")
    print(" 💤 TO DO   !!!!!!!!!!!!!! \n")
    modele = None
    print("🏁 compile_model() done \n")
    print("✅ Model compiled")
    return model

def train_model(model, X=None, y=None):
    """
    Fit the model and return  model or tuple (fitted_model, history)
    """
    print("🎬 train_model starting ................\n")
    print(" 💤 TO DO   !!!!!!!!!!!!!! \n")
    metrics = None
    history = model.fit(X, y)
    print("🏁 train_model() done \n")

    return model, history


def evaluate_model (model, X=None, y=None):
    """
    Evaluate trained model performance on the dataset
    """

    print("🎬 evaluate_model starting ................\n")
    print(" 💤 TO DO   !!!!!!!!!!!!!! \n")
    metrics = model.evaluate(X, y)
    print("🏁 evaluate_model() done \n")

    return metrics
