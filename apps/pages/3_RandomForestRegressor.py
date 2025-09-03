import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
from medium.ml_logic.data import load_json_from_files

st.set_page_config(page_title="RandomForestRegressor Model Page")

st.title("RandomForestRegressor Model ")


df = load_json_from_files(
    X_filepath='raw_data/X_test.json',
    y_filepath='raw_data/y_test.csv',
    num_lines=100
)
df = df[df['domain'] == 'medium.com'].copy()


title = st.selectbox('Select Title',df['title'])


# url / author
st.write('The url is ', df[df['title'] == title]['url'].values[0])
st.write('The author is ', df[df['title'] == title]['author'].iloc[0]['twitter'])


url = 'http://0.0.0.0:8000' # 'https://boost-medium-docker-759226870731.europe-west1.run.app'


dict_params = {
    'model_name': 'Ridge_punct_removed_stopwords_removed_data_scaled',
    'title': df[df['title'] == title].to_json()
}

prediction = requests.post(url=f"{url}/predict", json=dict_params)

#Prediction
if st.button('Recommandation'):
    st.write('The prediction is',int(round(prediction.json()['recommandations'])))
    st.write('Numbers of recommandations:', int(round(np.expm1(prediction.json()['recommandations']))))
    st.write('Numbers of real recommandations:', int(round(np.expm1(df[df['title'] == title]['log1p_recommends']))))
