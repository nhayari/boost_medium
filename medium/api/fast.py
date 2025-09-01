import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from medium.interface.main import pred
from medium.ml_logic.registry import load_model
from medium.ml_logic.preprocessor import MediumPreprocessingPipeline
from medium.params import *
from medium.ml_logic.model import *

app = FastAPI()

app.state.model = load_model(model_name='Ridge')



# Allowing all middleware is optional, but good practice for dev purposes
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all methods
#     allow_headers=["*"],  # Allows all headers
# )


@app.get("/predict")
def predict(model_name:str, text: str ):
    """
    Make a single course prediction.
    Assumes `text` is provided by the user
    """
    app.state.model = load_model(model_name=model_name)
    model = app.state.model
    X_processed = MediumPreprocessingPipeline()
    y_pred = model.predict(X_processed)
    return {'recommandations': y_pred}

@app.get("/")
def root():
    return {'To do ': 'Home Page'}



@app.get("/ping")
def ping():
    return "pong"
