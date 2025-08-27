# import glob
# import os
import time
import pickle

# from colorama import Fore, Style
from tensorflow import keras
# from google.cloud import storage

from medium.params import *
import mlflow
from mlflow.tracking import MlflowClient

def save_results(params: dict, metrics: dict) -> None:
    """
    Persist params & metrics locally on the hard drive at
    "{LOCAL_REGISTRY_PATH}/params/{current_timestamp}.pickle"
    "{LOCAL_REGISTRY_PATH}/metrics/{current_timestamp}.pickle"
    - (unit 03 only) if MODEL_TARGET='mlflow', also persist them on MLflow
    """
    print("🎬 save_results starting ................\n")
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save params locally
    if params is not None:
        params_path = os.path.join(LOCAL_REGISTRY_PATH, "params", timestamp + ".pickle")
        with open(params_path, "wb") as file:
            pickle.dump(params, file)

    # Save metrics locally
    if metrics is not None:
        metrics_path = os.path.join(LOCAL_REGISTRY_PATH, "metrics", timestamp + ".pickle")
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    print(" 💤 TO DO   !!!!!!!!!!!!!! \n")
    print("🏁 save_results() done \n")



def save_model(model: keras.Model = None) -> None:
    """
    Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}"
    - if MODEL_TARGET='gcs', also persist it in bucket on GCS
    - if MODEL_TARGET='mlflow', also persist it on MLflow instead of GCS
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
    - or from MLFLOW (by "stage") if MODEL_TARGET=='mlflow'

    Return None (but do not Raise) if no model is found
    """
    print("🎬 load_model starting ................\n")
    print(" 💤 TO DO   !!!!!!!!!!!!!! \n")
    print("🏁 load_model() done \n")

    return None



def mlflow_transition_model(current_stage: str, new_stage: str) -> None:
    """
    Transition the latest model from the `current_stage` to the
    `new_stage` and archive the existing model in `new_stage`
    """
    print("🎬 mlflow_transition_model starting ................\n")
    print(" 💤 TO DO   !!!!!!!!!!!!!! \n")
    print("🏁 mlflow_transition_model() done \n")
    return None


def mlflow_run(func):
    """
    Generic function to log params and results to MLflow along with TensorFlow auto-logging

    Args:
        - func (function): Function you want to run within the MLflow run
        # - params (dict, optional): Params to add to the run in MLflow. Defaults to None.
        # - context (str, optional): Param describing the context of the run. Defaults to "Train".
    """
    print("🎬 mlflow_run starting ................\n")
    print(" 💤 TO DO   !!!!!!!!!!!!!! \n")
    # def wrapper(*args, **kwargs):
    #     mlflow.end_run()
    #     mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    #     mlflow.set_experiment(experiment_name=MLFLOW_EXPERIMENT)

    #     with mlflow.start_run():
    #         mlflow.tensorflow.autolog()
    #         results = func(*args, **kwargs)

    #     print("✅ mlflow_run auto-log done")

    #     return results
    # return wrapper
    print("🏁 mlflow_run end \n")
    return None
