import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
from medium.ml_logic.data import load_json_from_files

st.set_page_config(page_title="RandomForestRegressor Model Page")


st.markdown(
    """
    <h1 style='text-align: center; color: white;'>🌴 RandomForestRegressor Model </h1>
    """,
    unsafe_allow_html=True
)


df = load_json_from_files(
    X_filepath='raw_data/X_test.json',
    y_filepath='raw_data/y_test.csv',
    num_lines=100
)
df = df[df['domain'] == 'medium.com'].copy()


df_title_selected = ['Are you a journalist? Download this free guide for verifying photos and videos',
                     "How To Grow Your Company: Don’t Use Facebook – Rodrigo Tello – Medium",
                     "Ideas to start saving for your own Round-The-World Trip or just about anything",
                     "Jesus Goes to Africa – The Bigger Picture – Medium",
                     "200+ Podcasts, 100+ Articles, 20+ Books… In 11 Bullet Points",
                     "We’re seeking design thinkers, talented tinkerers and wannabe surfers…"
]

title = st.selectbox('Select Title',df_title_selected)

# url / author
st.write('The url is ', df[df['title'] == title]['url'].values[0])
st.write('The author is ', df[df['title'] == title]['author'].iloc[0]['twitter'])


url = 'http://0.0.0.0:8000' # 'https://boost-medium-docker-759226870731.europe-west1.run.app'


dict_params = {
    'model_name': 'RandomForest',
    'medium': df[df['title'] == title].to_json()
}

prediction = requests.post(url=f"{url}/predict", json=dict_params)

#Prediction
if st.button("📊 Show Number of Claps"):
    st.write('**👏 Claps predicted:**', int(round(np.expm1(prediction.json()['recommandations']))))
    st.write('**✅ Real claps on extraction:**', int(round(np.expm1(df[df['title'] == title]['log1p_recommends']))))