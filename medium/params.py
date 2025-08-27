import os
import numpy as np

##################  VARIABLES  ##################
DATA_TRAIN = os.environ.get("DATA_TRAIN")
DATA_TEST = os.environ.get("DATA_TEST")
DATA_LOG_RECOMMEND = os.environ.get("DATA_LOG_RECOMMEND")
MODEL_TARGET = os.environ.get("MODEL_TARGET")

# LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), "medium", "data")
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), "medium")
