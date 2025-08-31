import math
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from typing import List, Optional


class BooleanEncoder(BaseEstimator, TransformerMixin):
    """Encode boolean-like values to 0/1."""
    
    def __init__(self, columns: Optional[List[str]] = None):
        self.columns = columns or []
    
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform boolean-like columns to 0/1."""
        X_transformed = X.copy()
        
        for col in self.columns:
            if col in X_transformed.columns:
                X_transformed[col] = X_transformed[col].fillna(0).apply(lambda x: 1 if x else 0)
        
        return X_transformed


class DomainEncoder(BaseEstimator, TransformerMixin):
    """Encode domain column to check if it's medium.com."""
    
    def __init__(self, domain_column: str = 'domain', target_domain: str = 'medium.com'):
        self.domain_column = domain_column
        self.target_domain = target_domain
    
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform domain column to binary encoding."""
        X_transformed = X.copy()
        
        if self.domain_column in X_transformed.columns:
            X_transformed[self.domain_column] = (
                X_transformed[self.domain_column]
                .fillna('')
                .str.lower() == self.target_domain.lower()
            ).astype(int)
        
        return X_transformed


class ReferrerEncoder(BaseEstimator, TransformerMixin):
    """Encode referrer column to check if it's origin."""
    
    def __init__(self, referrer_column: str = 'meta_tags_referrer', target_referrer: str = 'origin'):
        self.referrer_column = referrer_column
        self.target_referrer = target_referrer
    
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform referrer column to binary encoding."""
        X_transformed = X.copy()
        
        if self.referrer_column in X_transformed.columns:
            X_transformed[self.referrer_column] = (
                X_transformed[self.referrer_column].fillna('') == self.target_referrer
            ).astype(int)
        
        return X_transformed


class RobotsEncoder(BaseEstimator, TransformerMixin):
    """Encode robots column to check if it's 'index, follow'."""
    
    def __init__(self, robots_column: str = 'meta_tags_robots', target_robots: str = 'index, follow'):
        self.robots_column = robots_column
        self.target_robots = target_robots
    
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform robots column to binary encoding."""
        X_transformed = X.copy()
        
        if self.robots_column in X_transformed.columns:
            X_transformed[self.robots_column] = (
                X_transformed[self.robots_column].fillna('') == self.target_robots
            ).astype(int)
        
        return X_transformed


# Backward compatibility functions
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
