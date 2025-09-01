from medium.deep_learning.data import DeepLearningData


class Medium:

    def __init__(self):
        self.data = DeepLearningData().load_data()
        self.preprocessed_data = DeepLearningData().load_preprocess_data(self.data)

    def load_preprocess_data(self):


    def preprocess_data(self):
        """
        Preprocess the loaded data for training.
        """
        # Implement your preprocessing steps here
        pass
