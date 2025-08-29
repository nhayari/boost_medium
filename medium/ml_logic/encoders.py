import math
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder

def encode_zero_ones(col: pd.Series):
    return col.apply(lambda x: 1 if x else 0)

def encode_domain(col: pd.Series):
    return (col == 'medium.com').astype(int)

def encode_referrer(col: pd.Series):
    return (col == 'origin').astype(int)

def encode_robots(col: pd.Series):
    return (col == 'index, follow').astype(int)
