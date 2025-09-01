import streamlit as st
import pandas as pd
import numpy as np
from medium.ml_logic.registry import load_model
from medium.ml_logic.preprocessor import preprocess_features
import time
import datetime
import requests
from medium.api.fast import predict
from medium.ml_logic.model import *

st.markdown("""
    # Boost Medium Articles App
""")
# CSS = """
# .stApp {
#     background-image: url(....);
#     background-size: cover;
# }
# """


# Vérifier la connexion
def check_connection():
    return time.time() % 10 > 5  # connecté si le modulo > 5

st.title("Indicateur de connexion")

# Bouton cliquable
if st.button("🔄 Vérifier la connexion"):
    connected = check_connection()

    if connected:
        st.markdown(
            "<div style='width:20px;height:20px;border-radius:50%;background-color:green;'></div>",
            unsafe_allow_html=True
        )
        st.success("You're connected ! ✅")
    else:
        st.markdown(
            "<div style='width:20px;height:20px;border-radius:50%;background-color:red;'></div>",
            unsafe_allow_html=True
        )
        st.error("You're not connected! ❌")
else:
    st.info("Clique sur le bouton pour vérifier ta connexion 🚀")



# Sélection du modèle
list_model = st.selectbox('Select Model', ['Ridge','GradientBoostingRegressor'])

# implemented_model=list(implemented_model.keys().range(0,9))
# option = st.selectbox('Select Model', implemented_model)
# list_model = [implemented_model == option]


# options = list(implemented_model.values())[:9]

# # Selectbox avec les valeurs
# option = st.selectbox("Select Model", options)

#Dates
date = st.date_input(
    "Date",
    datetime.date(2025, 5, 6))



#Titre
title = st.text_input('Title', 'Your title here')


#Récupération du contenu
if st.button ('Generate content of the title'):
    st.write('content of the title')

url = 'https://boost-medium-docker-759226870731.europe-west1.run.app/predict'


dict_params = {
    'model_name': [list_model],
    'text': title
}


prediction = requests.get(url=url, params=dict_params)


#Prediction
if st.button('Prediction'):
    st.write('The prediction is ',prediction)
