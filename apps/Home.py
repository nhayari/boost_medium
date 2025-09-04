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

# Set background color
st.markdown(
    """
    <h1 style='text-align: center; color: white;'>🚀 Boost Medium Articles App</h1>
    """,
    unsafe_allow_html=True
)

# Display image
st.markdown(
    """
    <div style='text-align: center;'>
        <img src='https://raw.githubusercontent.com/nhayari/boost_medium/refs/heads/dev/apps/medium.png' width='300'>


    </div>
    """,
    unsafe_allow_html=True
)

# Intro text
st.markdown(
    """
    <div style="text-align: left; font-size:18px; line-height:1.6;">
    Welcome to our prediction app, powered by <b>5 complementary models</b>:
    </div>
    """,
    unsafe_allow_html=True
)

# List of models
st.markdown("""
- A **Deep Learning model (CNN)**,
- And four **Machine Learning models**: **Ridge**, **ExtraTreesRegressor**, **GradientBoostingRegressor**, and **RandomForestRegressor**.
This diverse set of approaches ensures robustness, accuracy, and adaptability to the data.
""")

# Additional info
st.markdown(
    """
    <div style="text-align: left; font-size:18px; line-height:1.6;">
    Our app is designed to help you predict the number of claps your Medium articles might receive.
    Whether you're a seasoned writer or just starting out, our models can provide valuable insights to enhance your content strategy.
    </div>
    """,
    unsafe_allow_html=True
)
