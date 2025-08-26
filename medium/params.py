import os
import numpy as np

##################  VARIABLES  ##################
DATA_TRAIN = os.environ.get("DATA_TRAIN")
DATA_TEST = os.environ.get("DATA_TEST")
DATA_LOG_RECOMMEND = os.environ.get("DATA_LOG_RECOMMEND")
MODEL_TARGET = os.environ.get("MODEL_TARGET")
MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME")

MLFLOW_TRACKING_URI = os.environ.get("https://mlflow.lewagon.ai")
MLFLOW_EXPERIMENT = os.environ.get("medium_experiment_nmlf")
MLFLOW_MODEL_NAME = os.environ.get("medium_mlf")

PREFECT_FLOW_NAME = os.environ.get("medium_lifecycle_pf")
PREFECT_LOG_LEVEL = os.environ.get("WARNING")

# CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE"))
# GCP_PROJECT = os.environ.get("GCP_PROJECT")
# GCP_REGION = os.environ.get("GCP_REGION")
# # Cloud Storage
# BUCKET_NAME=os.environ.get("BUCKET_NAME")
# # Compute Engine
# INSTANCE=os.environ.get("INSTANCE")


# LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), "medium", "data")
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), "medium")
