import streamlit as st
import pandas as pd
import numpy as np
from medium.ml_logic.registry import load_model
from medium.ml_logic.preprocessor import preprocess_features


st.markdown("""
    # Boost Medium Articles App
""")

# CSS = """
# .stApp {
#     background-image: url(https://raw.githubusercontent.com/nhayari/boost_medium/refs/heads/streamlit/apps/medium.webp) !important;
#     background-position: top center;
#     background-repeat: no-repeat;
#     background-size: 300px auto;
# }
# """
# st.markdown(f'<style>{CSS}</style>', unsafe_allow_html=True)


st.selectbox('Select Model', [' ','LinearRegression', 'RandomForestRegressor', 'ExtraTreesRegressor', 'ElasticNet'])
st.button("Evaluate Model")


if st.button('click me'):
    st.success("You're connected! 🎉")
    st.error("You're not connected! 😭")

import time

# Exemple : variable qui indique si on est connecté ou pas
# (dans ton cas, ça peut être un test de connexion à une API, BDD, etc.)
def check_connection():
    # Ici on simule la connexion (remplace par ta propre logique)
    return time.time() % 10 > 5  # connecté si le modulo > 5

st.title("Indicateur de connexion")

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
