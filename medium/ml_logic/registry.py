# import glob
# import os
# import time
# import pickle

# from colorama import Fore, Style
# from tensorflow import keras
# from google.cloud import storage

from medium.params import *

def save_results(params: dict, metrics: dict) -> None:
    """
    Persist params & metrics locally on the hard drive at
    "{LOCAL_REGISTRY_PATH}/params/{current_timestamp}.pickle"
    "{LOCAL_REGISTRY_PATH}/metrics/{current_timestamp}.pickle"
    """
    print("🎬 save_results starting ................\n")
    print(" 💤 TO DO   !!!!!!!!!!!!!! \n")
    print("🏁 save_results() done \n")



def save_model(model: keras.Model = None) -> None:
    """
    Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}"
    - if MODEL_TARGET='gcs', also persist it in bucket on GCS
    """
    print("🎬 save_model starting ................\n")
    print(" 💤 TO DO   !!!!!!!!!!!!!! \n")
    print("🏁 save_model() done \n")

    return None


def load_model(stage="Production") -> keras.Model:
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'

    Return None (but do not Raise) if no model is found
    """
    print("🎬 load_model starting ................\n")
    print(" 💤 TO DO   !!!!!!!!!!!!!! \n")
    print("🏁 load_model() done \n")

    return None
