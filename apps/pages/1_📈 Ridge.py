import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
from medium.ml_logic.data import load_json_from_files

st.set_page_config(page_title="Ridge Model Page", page_icon="📈")

st.title("Ridge Model ")


df = load_json_from_files(
    X_filepath='raw_data/test.json',
    y_filepath='raw_data/test_log1p_recommends.csv',
    num_lines=100
)

# Sélection du modèle
title = st.selectbox('Select Title', df['title'])


# url / author
st.write('The url is ', df[df['title'] == title]['url'].values[0])
st.write('The author is ', df[df['title'] == title]['author'].iloc[0]['twitter'])


url = 'http://0.0.0.0:8000' # 'https://boost-medium-docker-759226870731.europe-west1.run.app'


dict_params = {
    'model_name': 'Ridge',
    'title': title
}



# prediction = requests.get(url=f"{url}/predict", params=dict_params)

prediction = requests.post(url=f"{url}/predict", json=dict_params)


#Prediction
if st.button('Prediction'):
    st.write('The prediction is ',prediction)
