import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
from medium.ml_logic.data import load_json_from_files

st.set_page_config(page_title="Ridge Model Page", page_icon="📈")

df = load_json_from_files(
    X_filepath='raw_data/test.json',
    y_filepath='raw_data/test_log1p_recommends.csv',
    num_lines=100
)

# Sélection du modèle
title = st.selectbox('Select Title', df['title'])

# url / author



url = 'https://boost-medium-docker-759226870731.europe-west1.run.app'


dict_params = {
    'model_name': 'Ridge'
}



prediction = requests.get(url=url, params=dict_params)


#Prediction
if st.button('Prediction'):
    st.write('The prediction is ',prediction)
