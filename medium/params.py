import os
import numpy as np

##################  VARIABLES  ##################
DATA_TRAIN = os.environ.get("DATA_TRAIN")
DATA_TEST = os.environ.get("DATA_TEST")
DATA_LOG_RECOMMEND = os.environ.get("DATA_LOG_RECOMMEND")
MODEL_TARGET = os.environ.get("MODEL_TARGET")
# CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE"))
# GCP_PROJECT = os.environ.get("GCP_PROJECT")
# GCP_REGION = os.environ.get("GCP_REGION")
# # Cloud Storage
# BUCKET_NAME=os.environ.get("BUCKET_NAME")
# # Compute Engine
# INSTANCE=os.environ.get("INSTANCE")
