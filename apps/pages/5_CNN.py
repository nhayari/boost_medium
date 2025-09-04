import streamlit as st
import pandas as pd
import numpy as np
import requests

from data import get_author

st.set_page_config(page_title="CNN Model Page")

st.markdown(
    """
    <h1 style='text-align: center; color: white;'>🧊 CNN Model </h1>
    """,
    unsafe_allow_html=True
)

df = pd.read_parquet('apps/mediums.parquet')


df_title_selected = ['Are you a journalist? Download this free guide for verifying photos and videos',
                     "How To Grow Your Company: Don’t Use Facebook – Rodrigo Tello – Medium",
                     "Ideas to start saving for your own Round-The-World Trip or just about anything",
                     "Jesus Goes to Africa – The Bigger Picture – Medium",
                     "200+ Podcasts, 100+ Articles, 20+ Books… In 11 Bullet Points",
                     "We’re seeking design thinkers, talented tinkerers and wannabe surfers…"
]


title = st.selectbox('Select Title',df_title_selected)

selected_medium = df[df['title'] == title]
selected_medium['author'] = selected_medium.apply(get_author, axis=1)


# url / author
st.write('The url is ', selected_medium['url'].values[0])
st.write('The author is ', selected_medium['author'].values[0])


url = st.secrets["medium_api_url"]


dict_params = {
    'model_name': 'CNN',
    'medium': df[df['title'] == title].to_json()
}


prediction = requests.post(url=f"{url}/predict/CNN", json=dict_params)

if st.button("📊 Show Number of Claps"):
    st.write('**👏 Claps predicted:**', prediction.json()['claps'])
    st.write('**✅ Real claps on extraction:**', int(round(np.expm1(df[df['title'] == title]['log1p_recommends']))))
