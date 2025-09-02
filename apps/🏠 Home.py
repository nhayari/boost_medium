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


st.set_page_config(page_title="Home Page", page_icon="🏠")


st.title('Boost Medium Articles App')


CSS = """
.stApp {
    background-image: url('medium.png');
    background-size: cover;
}
"""

# Vérifier la connexion
def check_connection():
    return time.time() % 10 > 5  # connecté si le modulo > 5

st.header("Indicateur de connexion")

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
