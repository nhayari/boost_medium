import os
import numpy as np

##################  VARIABLES  ##################
DATA_SIZE = int(os.environ.get("DATA_SIZE", 50000))
DATA_TEST_SIZE = int(os.environ.get("DATA_TEST_SIZE", 30000))
DATA_TRAIN = os.environ.get("DATA_TRAIN","raw_data/train.json")
DATA_TEST = os.environ.get("DATA_TEST","raw_data/test.json")
DATA_LOG_RECOMMEND = os.environ.get("DATA_LOG_RECOMMEND","raw_data/train_log1p_recommends.csv")
DATA_TEST_LOG_RECOMMEND = os.environ.get("DATA_TEST_LOG_RECOMMEND","raw_data/test_log1p_recommends.csv")
MODEL_TARGET = os.environ.get("MODEL_TARGET")

# LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), "medium", "data")
# LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), "medium")

PATH_DATA=os.path.join(os.path.expanduser('~'), "medium", "data")
PATH_PARAMAS=os.path.join(os.path.expanduser('~'), "medium", "params")
PATH_METRICS=os.path.join(os.path.expanduser('~'), "medium", "metrics")
PATH_MODELS=os.path.join(os.path.expanduser('~'), "medium", "models")
PATH_PREPROCESSOR=os.path.join(os.path.expanduser('~'), "medium", "preprocessor")
PATH_PREDICTION=os.path.join(os.path.expanduser('~'), "medium", "prediction")
