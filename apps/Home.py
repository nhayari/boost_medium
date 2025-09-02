import streamlit as st
import pandas as pd
import numpy as np
from medium.ml_logic.registry import load_model
import time
import datetime
import requests
from medium.api.fast import predict
from medium.ml_logic.model import *


st.set_page_config(page_title="Home Page", page_icon="🏠")


st.markdown(
    """
    <div style='text-align: center;'>
        <h1>Boost Medium Articles App</h1>
        <img src='https://raw.githubusercontent.com/nhayari/boost_medium/refs/heads/dev/apps/medium.png' width='300'>


    </div>
    """,
    unsafe_allow_html=True
)


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
