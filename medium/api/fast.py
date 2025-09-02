import pandas as pd
from fastapi import FastAPI, Request
from medium.ml_logic.registry import load_model,load_preprocessor
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
    model = load_model(model_identifier=request_data['model_name'])
    df_title = pd.DataFrame(request_data['title'])
    y_pred = model.predict(df_title)
    print(np.expm1(y_pred))
    return {'recommandations': float(y_pred)}

@app.get("/")
def root():
    return {'To do ': 'Home Page'}



@app.get("/ping")
def ping():
    return "pong"
