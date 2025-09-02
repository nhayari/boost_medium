import pickle
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
            print(f"🎬 save_model starting {model_name}................\n")
            timestamp = time.strftime("%Y%m%d-%H%M%S")

            model_path = os.path.join(self.local_model_directory, f"{model_name}_{DATA_SIZE}_{timestamp}.keras")
            model.save(model_path)
            print(f" ✅ save_model() done \n")
            return True
        except Exception as e:
            print(f" 🛑  save_model() failed: {e}")
            return False


    def load_model(self, model_name):
        local_model_paths = glob.glob(f"{self.local_model_directory}/{model_name}_{DATA_SIZE}_*.keras")
        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]
        model = keras.models.load_model(most_recent_model_path_on_disk)
        self.register_model(model_name, model)

        return model

class PreprocessorRegistry():
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(PreprocessorRegistry, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.preprocessors = {}

    def register_preprocessor(self, name, preprocessor):
        self.preprocessors[name] = preprocessor

    def get_preprocessor(self, name):
        return self.preprocessors.get(name)

    def remove_preprocessor(self, name):
        if name in self.preprocessors:
            del self.preprocessors[name]

    def save_preprocessor(self, preprocessor, name:str='preprocessor') -> bool:
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


    def load_preprocessor(self, name:str='preprocessor'):
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
