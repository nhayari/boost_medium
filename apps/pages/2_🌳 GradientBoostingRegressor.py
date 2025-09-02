import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
from medium.ml_logic.data import load_json_from_files

st.set_page_config(page_title="GradientBoostingRegressor Model Page", page_icon="🌳")

st.title("GradientBoostingRegressor Model ")


df = load_json_from_files(
    X_filepath='raw_data/X_test.json',
    y_filepath='raw_data/y_test.csv',
    num_lines=100
)

# Sélection du modèle
title = st.selectbox('Select Title', df['title'])


# url / author
st.write('The url is ', df[df['title'] == title]['url'].values[0])
st.write('The author is ', df[df['title'] == title]['author'].iloc[0]['twitter'])


url = 'https://boost-medium-docker-759226870731.europe-west1.run.app'


dict_params = {
    'model_name': 'GradientBoostingRegressor_punct_removed_stopwords_removed',
    'title': df[df['title'] == title].to_json()
}


prediction = requests.get(url=url, params=dict_params)


#Prediction
if st.button('Prediction'):
    st.write(prediction)
