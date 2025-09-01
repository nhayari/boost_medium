from tensorflow import keras
from medium.params import *

import glob
import os
import time

class ModelRegistry():
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.models = {}
        self.local_model_directory = os.path.join(PATH_MODELS,"deep_learning")

    def register_model(self, name, model):
        self.models[name] = model

    def get_model(self, name):
        return self.models.get(name)

    def remove_model(self, name):
        if name in self.models:
            del self.models[name]

    def save_model(self, model_name, model):
        try:
            print("🎬 save_model starting ................\n")
            timestamp = time.strftime("%Y%m%d-%H%M%S")

            model_path = os.path.join(self.local_model_directory, f"{model_name}_{DATA_SIZE}_{timestamp}.h5")
            model.save(model_path)
            print(f" ✅ save_model() done \n")
            return True
        except Exception as e:
            print(f" 🛑  save_model() failed: {e}")
            return False


    def load_model(self, model_name):
        local_model_paths = glob.glob(f"{self.local_model_directory}/{model_name}_{DATA_SIZE}_*.h5")

        print(f" ℹ️  chemin des models : {self.local_model_directory} ")
        if not local_model_paths:
            print(f" 🛑  aucun model chargé !!")
            return None

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]
        model = keras.models.load_model(most_recent_model_path_on_disk)
        self.register_model(model_name, model)

        return model
