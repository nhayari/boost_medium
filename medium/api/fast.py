import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from medium.interface.main import pred
from medium.ml_logic.registry import load_model,load_preprocessor
from medium.ml_logic.preprocessor import MediumPreprocessingPipeline
from medium.params import *
from medium.ml_logic.model import *
import json

app = FastAPI()

# app.state.model = load_model(model_name='Ridge')



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
    request_data['title'] = json.loads(request_data['title'])
    print(type(request_data['title']))
    model = load_model(model_name=request_data['model_name'])
    df_title = pd.DataFrame(request_data['title'])
    X_proc = load_preprocessor('medium_pipeline_preprocessor')
    X_processed = X_proc.fit_transform(df_title)
    y_pred = model.predict(X_processed)
    return {'recommandations': float(y_pred)}

@app.get("/")
def root():
    return {'To do ': 'Home Page'}



@app.get("/ping")
def ping():
    return "pong"
