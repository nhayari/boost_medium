import os
import numpy as np

##################  VARIABLES  ##################
DATA_SIZE = int(os.environ.get("DATA_SIZE"))
DATA_TRAIN = os.environ.get("DATA_TRAIN")
DATA_TEST = os.environ.get("DATA_TEST")
DATA_LOG_RECOMMEND = os.environ.get("DATA_LOG_RECOMMEND")
MODEL_TARGET = os.environ.get("MODEL_TARGET")
MODEL_TYPE = os.environ.get("MODEL_TYPE")

# LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), "medium", "data")
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), "medium")
