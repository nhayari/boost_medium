import math
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder

def encode_zero_ones(col: pd.Series):
    """Safely encode boolean-like values to 0/1"""
    return col.fillna(0).apply(lambda x: 1 if x else 0)

def encode_domain(col: pd.Series):
    """Safely encode domain column"""
    # Fill NaN with empty string to avoid issues
    return (col.fillna('').str.lower() == 'medium.com').astype(int)

def encode_referrer(col: pd.Series):
    """Safely encode referrer column"""
    return (col.fillna('') == 'origin').astype(int)

def encode_robots(col: pd.Series):
    """Safely encode robots column"""
    return (col.fillna('') == 'index, follow').astype(int)
