import sys
import pandas as pd

from medium.params import *

from medium.deep_learning.data import DeepLearningData
from medium.deep_learning.medium import Medium
from medium.deep_learning.registry import ModelRegistry, PreprocessorRegistry

def predict():
    model = ModelRegistry().load_model("CNN")
    medium = Medium(model=model)
    prediction_data = {
        'full_content': 'This is an example content for prediction.'
    }

    df_pred = pd.DataFrame([prediction_data])

    medium.tokenize_and_pad(df_pred, 'X_pred')

    predictions = medium.predict(medium.X_pred_pad)
    print(f"Predictions: {predictions}")


if __name__ == "__main__":
    try:
        if len(sys.argv) > 1 and sys.argv[1] == "predict":
            predict()
        else:
            # Initialize the Medium class
            medium = Medium()

            # Load and preprocess the data
            medium.load_data()

            # Tokenize and pad sequences :
            medium.tokenize_and_pad(medium.X_train, 'X_train')
            medium.tokenize_and_pad(medium.X_val, 'X_val')

            # Build and train the model
            medium.build_CNN_model()
            medium.fit()

            # Save the model
            model_name = medium.get_model_name()
            model_registry = ModelRegistry()
            model_registry.save_model(model_name, medium.model)

            # Evaluate the model
            medium.load_test_data()
            medium.tokenize_and_pad(medium.X_test, 'X_test')
            res = medium.evaluate(medium.X_test_pad, medium.y_test)
            print(f"Test Evaluation Results: {res}")

    except Exception as e:
        print(f"Error occurred at line: {e.__traceback__.tb_lineno} \n File: {e.__traceback__.tb_frame.f_code.co_filename} \n Error Message: {e}")
        exit(1)
    except ValueError as ve:
        print(f"ValueError: {ve}")
        exit(1)
