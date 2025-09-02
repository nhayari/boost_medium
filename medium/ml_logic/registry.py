import glob
import os
import time
import pickle

# from colorama import Fore, Style
# from tensorflow import keras
# from google.cloud import storage

from medium.params import *


def save_results(model_name,params: dict, metrics: dict) -> bool:
    """
    Persist params & metrics locally on the hard drive at
    "{LOCAL_REGISTRY_PATH}/params/{current_timestamp}.pickle"
    "{LOCAL_REGISTRY_PATH}/metrics/{current_timestamp}.pickle"
    """
    try:
        print("🎬 save_results starting ................\n")
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        # Save params locally
        if params is not None:
            params_path = os.path.join(PATH_PARAMAS, f"{model_name}_{DATA_SIZE}_{timestamp}.pickle")
            with open(params_path, "wb") as file:
                pickle.dump(params, file)

        # Save metrics locally
        if metrics is not None:
            metrics_path = os.path.join(PATH_METRICS, f"{model_name}_{DATA_SIZE}_{timestamp}.pickle")
            with open(metrics_path, "wb") as file:
                pickle.dump(metrics, file)

        print(f" ✅ save_results() done \n")
    except Exception as e:
        print(f"Error saving results: {e}")
        return False

    return True


def save_model(model) -> bool:
    """
    Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}"
    - if MODEL_TARGET='gcs', also persist it in bucket on GCS
    """
    # Save model locally
    try:
        print("🎬 save_model starting ................\n")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        # model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{MODEL_TYPE}_{DATA_SIZE}_{timestamp}.pickle")
        model_path = os.path.join(PATH_MODELS, f"{model.__class__.__name__}_{DATA_SIZE}_{timestamp}.pickle")
        with open(model_path, "wb") as file:
            pickle.dump(model, file)
        print(f" ✅ save_model() done \n")
    except Exception as e:
        print(f"Error saving model: {e}")
        return False

    return True


def load_model(model_name: str):
    """
    Return a saved model:

    Return None (but do not Raise) if no model is found
    """
    try:
        print("🎬 load_model starting ................\n")

        # Get the latest model version name by the timestamp on disk
        local_model_directory = os.path.join(PATH_MODELS,"")
        print(f"{local_model_directory}/{model_name}_{DATA_SIZE}_*.pickle")
        local_model_paths = glob.glob(f"{local_model_directory}/{model_name}_{DATA_SIZE}_*.pickle")
        print(f" ℹ️  chemin des models : {local_model_directory} ")
        if not local_model_paths:
            print(f" 🛑  aucun model chargé !!")
            return None

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]
        print(f"Loading {most_recent_model_path_on_disk}...")
        try:
            with open(most_recent_model_path_on_disk, "rb") as file:
                model = pickle.load(file)
                print(f" ✅ Loaded {most_recent_model_path_on_disk}. \n")
                return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def save_preprocessor(preprocessor,name:str='preprocessor') -> bool:
    """
    Save the preprocessor object locally on the hard drive at
    "{PATH_PREPROCESSOR}/{current_timestamp}.pickle"
    """
    try:
        print("🎬 save_preprocessor starting ................\n")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        preprocessor_path = os.path.join(PATH_PREPROCESSOR,f"{name}_{DATA_SIZE}_{timestamp}.pickle")
        with open(preprocessor_path, "wb") as file:
            pickle.dump(preprocessor, file)
        print(f" ✅ save_preprocessor() done \n")
    except Exception as e:
        print(f"Error saving preprocessor: {e}")
        return False

    return True


def load_preprocessor(name:str='preprocessor'):
    """
    Load the preprocessor object from disk.
    """
    try:
        print("🎬 load_preprocessor starting ................\n")

        # Get the latest preprocessor version name by the timestamp on disk
        local_preprocessor_directory = os.path.join(PATH_PREPROCESSOR,"")
        local_preprocessor_paths = glob.glob(f"{local_preprocessor_directory}/{name}_{DATA_SIZE}_*.pickle")
        # à vérifier la récupération du dernier !!
        if not local_preprocessor_paths:
            return None

        most_recent_preprocessor_path_on_disk = sorted(local_preprocessor_paths)[-1]
        try:
            with open(most_recent_preprocessor_path_on_disk, "rb") as file:
                preprocessor = pickle.load(file)
                print(f"✅ Loaded preprocessor {name}. \n")
                return preprocessor
        except Exception as e:
            print(f"Error loading preprocessor: {e}")
            return None
    except Exception as e:
        print(f"Error loading preprocessor: {e}")
        return None
