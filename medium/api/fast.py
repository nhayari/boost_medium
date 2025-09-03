import pandas as pd
import json

from fastapi import FastAPI, Request

from medium.ml_logic.registry import load_model,load_preprocessor
from medium.params import *
from medium.ml_logic.model import *

from medium.deep_learning.data import DeepLearningData
from medium.deep_learning.medium import Medium
from medium.deep_learning.registry import ModelRegistry, PreprocessorRegistry



app = FastAPI()

app.state.models = {
    'ML': {
        'Ridge': load_model(model_identifier='Ridge_punct_removed_stopwords_removed_data_scaled'),
        'GradientBoostingRegressor': load_model(model_identifier='GradientBoostingRegressor_punct_removed_stopwords_removed'),
        'ExtraTreesRegressor': load_model(model_identifier='ExtraTreesRegressor_punct_removed_stopwords_removed'),
        'RandomForest': load_model(model_identifier='RandomForestRegressor_punct_removed_stopwords_removed')
    },
    'DL': {
        'CNN': ModelRegistry().get_model('CNN')
    }
}



# Allowing all middleware is optional, but good practice for dev purposes
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all methods
#     allow_headers=["*"],  # Allows all headers
# )


@app.post("/predict")
async def predict(request: Request):
    """
    Make a single course prediction.
    Assumes `text` is provided by the user
    """
    request_data = await request.json()
    request_data['medium'] = json.loads(request_data['medium'])
    model = app.state.models['ML'][request_data['model_name']]
    df_title = pd.DataFrame(request_data['medium'])
    y_pred = model.predict(df_title)

    return {
        'model' : request_data['model_name'],
        'log1p' : float(y_pred),
        'claps' : int(np.expm1(y_pred))
    }


@app.post("/predict/CNN")
async def predict_CNN(request: Request):
    """
    predict the medium claps with CNN
    """
    request_data = await request.json()
    medium_df = pd.DataFrame(json.loads(request_data['medium']))

    # Load the CNN model
    model = app.state.models['DL']['CNN']

    # Init Medium instance
    medium_instance = Medium(model=model)

    # Load preprocessor
    preprocessor = PreprocessorRegistry().get_preprocessor('medium_preprocessor')
    preprocess_medium_df = DeepLearningData(preprocessor=preprocessor).preprocess_for_prediction(medium_df)

    # Tokenize and pad sequences
    medium_instance.tokenize_and_pad(preprocess_medium_df, attr_prefix='X_pred')

    # Make prediction
    y_pred = medium_instance.predict(medium_instance.X_pred_pad)

    return {
        'model' : model.name,
        'log1p' : float(y_pred),
        'claps' : int(np.expm1(y_pred))
    }


@app.get("/")
def root():
    return {'To do ': 'Home Page'}



@app.get("/ping")
def ping():
    return "pong"
