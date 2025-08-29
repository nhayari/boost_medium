import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from medium.interface.main import pred
# from medium.ml_logic.registry import load_model
# from medium.ml_logic.preprocessor import preprocess_features

app = FastAPI()
# app.state.model = load_model()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/predict")
def predict(model_name:str, text: str ):
    """
    Make a single course prediction.
    Assumes `text` is provided by the user
    """
    return pred(model_name, text)


@app.get("/")
def root():
    return {'To do ': 'Home Page'}
