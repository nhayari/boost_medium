from html import unescape
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Optional

import pandas as pd
pd.options.mode.copy_on_write = True
import re

HTML_TAG_PATTERN = re.compile(r'<[^>]+>')
WHITESPACE_PATTERN = re.compile(r'\s+')

class HTMLTagStripper(BaseEstimator, TransformerMixin):
    """Strip HTML tags from text columns."""

    def __init__(self, text_columns: Optional[List[str]] = None, chunk_size: int = 1000, show_progress: bool = True):
        self.text_columns = text_columns or ['content']
        self.chunk_size = chunk_size
        self.show_progress = show_progress

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Strip HTML tags from specified columns."""
        X_transformed = X.copy()

        def strip_tags_regex_compiled(html):
            """Use pre-compiled regex patterns for maximum speed."""
            if pd.isna(html):
                return html

            # Remove HTML tags with compiled regex
            clean = HTML_TAG_PATTERN.sub('', html)
            # Decode HTML entities
            clean = unescape(clean)
            # Clean up whitespace with compiled regex
            clean = WHITESPACE_PATTERN.sub(' ', clean).strip()

            return clean

        for text_col in self.text_columns:
            if text_col in X_transformed.columns:
                if self.show_progress:
                    try:
                        from tqdm import tqdm
                        progress_bar = tqdm(total=len(X_transformed[text_col]), desc=f"Stripping HTML tags from {text_col}")
                    except ImportError:
                        print("Install tqdm for progress tracking: pip install tqdm")
                        self.show_progress = False

                # Process in chunks to manage memory
                series = X_transformed[text_col]
                for i in range(0, len(series), self.chunk_size):
                    end_idx = min(i + self.chunk_size, len(series))

                    # Get chunk
                    chunk_mask = series.iloc[i:end_idx].notna()
                    if chunk_mask.any():
                        # Apply cleaning only to non-null values in chunk
                        chunk_indices = series.iloc[i:end_idx][chunk_mask].index
                        for idx in chunk_indices:
                            series.loc[idx] = strip_tags_regex_compiled(series.loc[idx])

                    if self.show_progress and 'progress_bar' in locals():
                        progress_bar.update(end_idx - i)

                if self.show_progress and 'progress_bar' in locals():
                    progress_bar.close()

                X_transformed[text_col] = series

        return X_transformed

class MediumPreprocessingPipeline(BaseEstimator, TransformerMixin):
    """Pipeline for medium text preprocessing."""

    def __init__(self, text_columns: Optional[List[str]] = None):
        self.text_columns = text_columns or ['content']
        self.html_stripper = HTMLTagStripper(text_columns=self.text_columns)

    def fit(self, X: pd.DataFrame, y=None):
        self.html_stripper.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self.html_stripper.transform(X)
        return X
