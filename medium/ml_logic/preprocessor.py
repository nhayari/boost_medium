import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import re
import nltk
from typing import List, Optional, Union
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from html import unescape
from textstat import flesch_reading_ease, flesch_kincaid_grade #type:ignore
from utils.text_preprocessing import remove_non_ascii, remove_punctuation, remove_stopwords, remove_extra_whitespace
from medium.ml_logic.data import clean_data
from medium.ml_logic.encoders import encode_referrer, encode_zero_ones
import warnings
warnings.filterwarnings('ignore')

HTML_TAG_PATTERN = re.compile(r'<[^>]+>')
WHITESPACE_PATTERN = re.compile(r'\s+')


class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract temporal features from datetime columns.
    FIXED: No longer uses future information (_timestamp).
    """

    def __init__(self, datetime_col: str = 'published_$date',
                 reference_date: str | None = None,
                 drop_original: bool = True):
        self.datetime_col = datetime_col
        self.reference_date = reference_date
        self.drop_original = drop_original

    def fit(self, X: pd.DataFrame, y=None):
        """
        During fit, we can establish a reference date based on training data.
        """
        if self.reference_date is None:
            # Use the maximum date in training data as reference
            # This simulates "now" being the latest date we have in training
            X_temp = X.copy()
            X_temp[self.datetime_col] = pd.to_datetime(X_temp[self.datetime_col])
            self.reference_date = X_temp[self.datetime_col].max()
            print(f"📅 Reference date set to: {self.reference_date}")
        else:
            self.reference_date = pd.to_datetime(self.reference_date)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal features from datetime column without using future information."""
        X_transformed = X.copy()

        if self.datetime_col not in X_transformed.columns:
            raise ValueError(f"Column '{self.datetime_col}' not found in DataFrame")

        X_transformed[self.datetime_col] = pd.to_datetime(X_transformed[self.datetime_col])

        # Basic temporal features
        X_transformed['publication_year'] = X_transformed[self.datetime_col].dt.year
        X_transformed['publication_month'] = X_transformed[self.datetime_col].dt.month
        X_transformed['publication_day'] = X_transformed[self.datetime_col].dt.day
        X_transformed['publication_dayofweek'] = X_transformed[self.datetime_col].dt.day_of_week
        X_transformed['publication_hour'] = X_transformed[self.datetime_col].dt.hour
        X_transformed['publication_is_weekend'] = X_transformed['publication_dayofweek'].isin([5, 6]).astype(int)

        # Days since publication using reference date (not _timestamp)
        if self.reference_date is not None:
            X_transformed['days_since_publication'] = (
                pd.to_datetime(self.reference_date) - X_transformed[self.datetime_col]
            ).dt.days #type:ignore
            # Clip negative values (articles published after reference date) to 0
            X_transformed['days_since_publication'] = X_transformed['days_since_publication'].clip(lower=0)

        if self.drop_original:
            X_transformed = X_transformed.drop(columns=[self.datetime_col])
            # Also drop _timestamp as it contains future information
            if '_timestamp' in X_transformed.columns:
                X_transformed = X_transformed.drop(columns=['_timestamp'])

        return X_transformed


def extract_temporal_features(df: pd.DataFrame, datetime_col: str,
                             reference_date: str | None = None,
                             drop: bool = False) -> pd.DataFrame:
    """
    Extract temporal features from a datetime column WITHOUT using future information.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing the datetime column.
    datetime_col : str
        Name of the datetime column to extract features from.
    reference_date : str or pd.Timestamp
        Reference date to calculate days_since_publication.
        If None, uses the max date in the dataframe.
    drop : bool
        Whether to drop the original datetime column after feature extraction.

    Returns:
    --------
    pandas.DataFrame
        Dataframe with new temporal features added.
    """
    df_temp = df.copy()
    df_temp[datetime_col] = pd.to_datetime(df_temp[datetime_col])

    # Basic temporal features
    df_temp['publication_year'] = df_temp[datetime_col].dt.year
    df_temp['publication_month'] = df_temp[datetime_col].dt.month
    df_temp['publication_day'] = df_temp[datetime_col].dt.day
    df_temp['publication_dayofweek'] = df_temp[datetime_col].dt.day_of_week
    df_temp['publication_hour'] = df_temp[datetime_col].dt.hour
    df_temp['publication_is_weekend'] = df_temp['publication_dayofweek'].isin([5, 6]).astype(int)

    # Calculate days_since_publication using reference date (not _timestamp)
    if reference_date is None:
        # Use max date in current data as reference
        reference_date = df_temp[datetime_col].max()
    else:
        reference_date = pd.to_datetime(reference_date) # type:ignore

    df_temp['days_since_publication'] = (reference_date - df_temp[datetime_col]).dt.days # type:ignore
    # Clip negative values to 0
    df_temp['days_since_publication'] = df_temp['days_since_publication'].clip(lower=0)

    if drop:
        df_temp = df_temp.drop(columns=[datetime_col])
        # Remove _timestamp as it contains future information
        if '_timestamp' in df_temp.columns:
            df_temp = df_temp.drop(columns=['_timestamp'])

    return df_temp

# Keep all other extractors the same...
def extract_text_features(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Extract basic text features from a text column in the dataframe."""
    df_text = df.copy()
    df_text[f'{text_col}_length'] = df_text[text_col].astype(str).apply(len)
    df_text[f'{text_col}_word_count'] = df_text[text_col].astype(str).apply(lambda x: len(x.split()))
    df_text[f'{text_col}_unique_word_count'] = df_text[text_col].astype(str).apply(lambda x: len(set(x.split())))
    df_text[f'{text_col}_has_numbers'] = df_text[text_col].astype(str).apply(lambda x: int(any(char.isdigit() for char in x)))
    df_text[f'{text_col}_is_question'] = df_text[text_col].astype(str).apply(lambda x: int(x.strip().endswith('?')))

    # Reading time
    if 'reading_time' not in df_text.columns and 'meta_tags_twitter:data1' in df_text.columns:
        df_text['reading_time'] = df_text['meta_tags_twitter:data1'].astype(str).str.extract(r'(\d+)').astype(float)

    return df_text


class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract basic text features from text columns."""

    def __init__(self, text_columns: Optional[List[str]] = None):
        self.text_columns = text_columns or ['title', 'content']

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract text features from specified columns."""
        X_transformed = X.copy()

        for text_col in self.text_columns:
            if text_col in X_transformed.columns:
                X_transformed[f'{text_col}_length'] = X_transformed[text_col].astype(str).apply(len)
                X_transformed[f'{text_col}_word_count'] = X_transformed[text_col].astype(str).apply(lambda x: len(x.split()))
                X_transformed[f'{text_col}_unique_word_count'] = X_transformed[text_col].astype(str).apply(lambda x: len(set(x.split())))
                X_transformed[f'{text_col}_has_numbers'] = X_transformed[text_col].astype(str).apply(lambda x: int(any(char.isdigit() for char in x)))
                X_transformed[f'{text_col}_is_question'] = X_transformed[text_col].astype(str).apply(lambda x: int(x.strip().endswith('?')))

        # Extract reading time if available
        if 'meta_tags_twitter:data1' in X_transformed.columns and 'reading_time' not in X_transformed.columns:
            X_transformed['reading_time'] = X_transformed['meta_tags_twitter:data1'].astype(str).str.extract(r'(\d+)').astype(float)

        return X_transformed


def extract_html_features(df: pd.DataFrame, html_col: str) -> pd.DataFrame:
    """Extract basic HTML features from a HTML content column in the dataframe."""
    df_html = df.copy()
    df_html[f'{html_col}_num_links'] = df_html[html_col].astype(str).apply(lambda x: len(re.findall(r'http[s]?://', x)))
    df_html[f'{html_col}_num_images'] = df_html[html_col].astype(str).apply(lambda x: len(re.findall(r'<img ', x)))
    df_html[f'{html_col}_num_lists'] = df_html[html_col].astype(str).apply(lambda x: len(re.findall(r'<ul|<ol', x)))
    df_html[f'{html_col}_num_paragraphs'] = df_html[html_col].astype(str).apply(lambda x: len(re.findall(r'<p', x)))
    df_html[f'{html_col}_num_h1'] = df_html[html_col].astype(str).apply(lambda x: len(re.findall(r'<h1', x)))
    df_html[f'{html_col}_num_h2'] = df_html[html_col].astype(str).apply(lambda x: len(re.findall(r'<h2', x)))
    df_html[f'{html_col}_num_h3'] = df_html[html_col].astype(str).apply(lambda x: len(re.findall(r'<h3', x)))

    return df_html


class HTMLFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract HTML features from HTML content columns."""

    def __init__(self, html_columns: Optional[List[str]] = None):
        self.html_columns = html_columns or ['content']

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract HTML features from specified columns."""
        X_transformed = X.copy()

        for html_col in self.html_columns:
            if html_col in X_transformed.columns:
                X_transformed[f'{html_col}_num_links'] = X_transformed[html_col].astype(str).apply(lambda x: len(re.findall(r'http[s]?://', x)))
                X_transformed[f'{html_col}_num_images'] = X_transformed[html_col].astype(str).apply(lambda x: len(re.findall(r'<img ', x)))
                X_transformed[f'{html_col}_num_lists'] = X_transformed[html_col].astype(str).apply(lambda x: len(re.findall(r'<ul|<ol', x)))
                X_transformed[f'{html_col}_num_paragraphs'] = X_transformed[html_col].astype(str).apply(lambda x: len(re.findall(r'<p', x)))
                X_transformed[f'{html_col}_num_h1'] = X_transformed[html_col].astype(str).apply(lambda x: len(re.findall(r'<h1', x)))
                X_transformed[f'{html_col}_num_h2'] = X_transformed[html_col].astype(str).apply(lambda x: len(re.findall(r'<h2', x)))
                X_transformed[f'{html_col}_num_h3'] = X_transformed[html_col].astype(str).apply(lambda x: len(re.findall(r'<h3', x)))

        return X_transformed


def strip_html_tags(series: pd.Series, chunk_size: int, show_progress: bool = True) -> pd.Series:
    """Memory-efficient processing with optional progress tracking."""

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

    if show_progress:
        try:
            from tqdm import tqdm
            progress_bar = tqdm(total=len(series), desc="Stripping HTML tags")
        except ImportError:
            print("Install tqdm for progress tracking: pip install tqdm")
            show_progress = False

    # Process in chunks to manage memory
    for i in range(0, len(series), chunk_size):
        end_idx = min(i + chunk_size, len(series))

        # Get chunk
        chunk_mask = series.iloc[i:end_idx].notna()
        if chunk_mask.any():
            # Apply cleaning only to non-null values in chunk
            chunk_indices = series.iloc[i:end_idx][chunk_mask].index
            for idx in chunk_indices:
                series.loc[idx] = strip_tags_regex_compiled(series.loc[idx])

        if show_progress and 'progress_bar' in locals():
            progress_bar.update(end_idx - i)

    if show_progress and 'progress_bar' in locals():
        progress_bar.close()

    return series


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


def extract_nlp_features(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Extract NLP features from a text column in the dataframe."""
    # Readability scores
    df['readability_score'] = df[f'{text_col}'].apply(flesch_reading_ease)
    df['grade_level'] = df[f'{text_col}'].apply(flesch_kincaid_grade)

    # Sentiment analysis
    df['title_sentiment'] = df['title'].apply(lambda x: TextBlob(x).sentiment.polarity) #type:ignore
    df['content_sentiment'] = df[f'{text_col}'].apply(
        lambda x: TextBlob(x[:5000]).sentiment.polarity if len(x) > 0 else 0 #type:ignore
    )

    return df


class NLPFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract NLP features like readability scores and sentiment analysis."""

    def __init__(self, text_columns: Optional[List[str]] = None):
        self.text_columns = text_columns or ['content']

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract NLP features from specified columns."""
        X_transformed = X.copy()

        for text_col in self.text_columns:
            if text_col in X_transformed.columns:
                # Readability scores
                X_transformed[f'{text_col}_readability_score'] = X_transformed[text_col].apply(flesch_reading_ease)
                X_transformed[f'{text_col}_grade_level'] = X_transformed[text_col].apply(flesch_kincaid_grade)

                # Sentiment analysis
                X_transformed[f'{text_col}_sentiment'] = X_transformed[text_col].apply(
                    lambda x: TextBlob(x[:5000]).sentiment.polarity if len(str(x)) > 0 else 0 #type:ignore
                )

        # Special case for title sentiment
        if 'title' in X_transformed.columns:
            X_transformed['title_sentiment'] = X_transformed['title'].apply(lambda x: TextBlob(str(x)).sentiment.polarity) #type:ignore

        return X_transformed


def clean_text(df: pd.DataFrame, text_col: str, drop_punctuation: bool,
               drop_stopwords: bool) -> pd.DataFrame:
    """Clean text data by lowercasing, removing punctuation, numbers, and extra whitespace."""

    def clean_single_text(text, drop_punctuation=drop_punctuation,
                        drop_stopwords=drop_stopwords):
        if not isinstance(text, str):
            return ""
        text = text.lower()

        if drop_punctuation:
            text = remove_punctuation(text)
        if drop_stopwords:
            text = remove_stopwords(text)

        text = remove_non_ascii(text)
        text = remove_extra_whitespace(text)
        return text

    df_cleaned = df.copy()
    df_cleaned[text_col] = df_cleaned[text_col].apply(clean_single_text)

    return df_cleaned


class TextCleaner(BaseEstimator, TransformerMixin):
    """Clean text data by lowercasing, removing punctuation, and stopwords."""

    def __init__(self, text_columns: Optional[List[str]] = None, drop_punctuation: bool = True, drop_stopwords: bool = True):
        self.text_columns = text_columns or ['content']
        self.drop_punctuation = drop_punctuation
        self.drop_stopwords = drop_stopwords

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clean text in specified columns."""
        X_transformed = X.copy()

        def clean_single_text(text):
            if not isinstance(text, str):
                return ""
            text = text.lower()

            if self.drop_punctuation:
                text = remove_punctuation(text)
            if self.drop_stopwords:
                text = remove_stopwords(text)

            text = remove_non_ascii(text)
            text = remove_extra_whitespace(text)
            return text

        for text_col in self.text_columns:
            if text_col in X_transformed.columns:
                X_transformed[text_col] = X_transformed[text_col].apply(clean_single_text)

        return X_transformed


def tokenize_and_lemmatize(text):
    """Tokenize and lemmatize text."""
    #tokenize
    tokens = word_tokenize(text)

    #lemmatize
    tokens_lemmatized = [WordNetLemmatizer().lemmatize(word)
                         for word in tokens
                         ]

    return (' '.join(tokens_lemmatized)).strip()


class TokenizerLemmatizer(BaseEstimator, TransformerMixin):
    """Tokenize and lemmatize text columns."""

    def __init__(self, text_columns: Optional[List[str]] = None, output_suffix: str = '_lemmatized'):
        self.text_columns = text_columns or ['content']
        self.output_suffix = output_suffix
        self.lemmatizer = None

    def fit(self, X: pd.DataFrame, y=None):
        # Initialize lemmatizer during fit
        self.lemmatizer = WordNetLemmatizer()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Tokenize and lemmatize text in specified columns."""
        X_transformed = X.copy()

        if self.lemmatizer is None:
            self.lemmatizer = WordNetLemmatizer()

        def tokenize_and_lemmatize_text(text):
            if not isinstance(text, str) or len(text.strip()) == 0:
                return ""

            # Tokenize
            tokens = word_tokenize(text)

            # Lemmatize
            tokens_lemmatized = [self.lemmatizer.lemmatize(word) for word in tokens] if self.lemmatizer else tokens

            return (' '.join(tokens_lemmatized)).strip()

        for text_col in self.text_columns:
            if text_col in X_transformed.columns:
                output_col = f'{text_col}{self.output_suffix}'
                X_transformed[output_col] = X_transformed[text_col].apply(tokenize_and_lemmatize_text)

        return X_transformed


def preprocess_pred(data: pd.DataFrame, preprocessor: any): #type:ignore
    """Preprocess prediction data."""
    print("🎬 preprocess_pred starting ................\n")

    data['text_lemmatized'] = data['content'].apply(tokenize_and_lemmatize)
    X_processed = preprocessor.transform(data['text_lemmatized'])
    X_processed = pd.DataFrame(X_processed.toarray(),
                    columns = preprocessor.get_feature_names_out())

    print("✅ preprocess_pred() done \n")

    return X_processed


class MediumPreprocessingPipeline(BaseEstimator, TransformerMixin):
    """
    Complete preprocessing pipeline for Medium articles data.
    FIXED: Temporal features no longer use future information.
    """

    def __init__(
        self,
        datetime_col: str = 'published_$date',
        text_columns: Optional[List[str]] = None,
        html_columns: Optional[List[str]] = None,
        chunk_size: int = 1000,
        remove_punct: bool = False,
        remove_stopwords: bool = False,
        tf_idf_min_ratio: float = 0.02,
        metadata_only: bool = False,
        content_only: bool = False,
        model_is_tree: bool = False,
        show_progress: bool = True,
        reference_date: str | None = None
    ):
        self.datetime_col = datetime_col
        self.text_columns = text_columns or ['title', 'content']
        self.html_columns = html_columns or ['content']
        self.chunk_size = chunk_size
        self.remove_punct = remove_punct
        self.remove_stopwords = remove_stopwords
        self.tf_idf_min_ratio = tf_idf_min_ratio
        self.show_progress = show_progress
        self.metadata_only = metadata_only
        self.content_only = content_only
        self.model_is_tree = model_is_tree
        self.reference_date = reference_date

        # Initialize transformers
        self.tfidf_vectorizer = None
        self.std_scaler = None
        self.feature_columns_ = None
        self.tfidf_feature_names_ = None
        self.temporal_extractor = None

    def fit(self, X: pd.DataFrame, y=None):
        """Fit the preprocessing pipeline on TRAINING DATA ONLY."""
        print("🎬 Fitting preprocessing pipeline on training data...")

        # Clean data first - keep track of indices for target alignment
        X_clean = clean_data(X.copy())

        # Remove unnecessary columns
        X_clean = self._remove_unnecessary_columns(X_clean)

        # Apply all transformations to learn parameters
        X_transformed = self._apply_transformations(X_clean, fit=True)

        # Store feature columns for consistency
        self.feature_columns_ = X_transformed.columns.tolist()
        if 'log1p_recommends' in self.feature_columns_:
            self.feature_columns_.remove('log1p_recommends')

        print("✅ Preprocessing pipeline fitted successfully on training data!")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted pipeline (for validation/test data)."""
        print("🔄 Transforming data with fitted preprocessing pipeline...")

        # Clean data first
        X_clean = clean_data(X.copy())

        # Remove unnecessary columns
        X_clean = self._remove_unnecessary_columns(X_clean)

        # Apply all transformations using fitted parameters
        X_transformed = self._apply_transformations(X_clean, fit=False)

        print("✅ Data transformation completed!")
        return X_transformed

    def _remove_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove unnecessary columns."""
        cols_to_keep = ['_timestamp',
            'title',
            'content',
            'image_url',
            'log1p_recommends',
            'published_$date',
            'link_tags_amphtml',
            'meta_tags_referrer',
            'meta_tags_twitter:data1'
        ]
        for col in cols_to_keep:
            if col in df.columns.to_list():
                pass
            else:
                df[col] = 0

        df = df[cols_to_keep].copy()

        if len(df)==1:
            df = df.drop(columns='log1p_recommends')
        return df

    def _apply_transformations(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Apply all transformations in sequence."""
        X = df.copy()

        # Store target variable if it exists
        y = None
        if 'log1p_recommends' in X.columns:
            y = X['log1p_recommends']
            X = X.drop(columns='log1p_recommends')

        numerical_cols = [
            'publication_year', 'publication_month', 'publication_day', 'publication_dayofweek',
            'publication_hour', 'days_since_publication', 'title_length', 'title_word_count',
            'title_unique_word_count', 'reading_time', 'content_length', 'content_word_count',
            'content_unique_word_count', 'content_has_numbers', 'content_is_question',
            'content_num_links', 'content_num_images', 'content_num_lists', 'content_num_paragraphs',
            'content_num_h1', 'content_num_h2', 'content_num_h3', 'content_readability_score',
            'content_grade_level'
        ]

        if self.content_only:
            print("Skipping metadata columns analysis. ")
            for col in numerical_cols + ['_timestamp','image_url','published_$date','link_tags_amphtml','meta_tags_referrer','meta_tags_twitter:data1']:
                if col in X.columns:
                    X = X.drop(columns = col)
        else:
            # 1. Extract temporal features (FIXED: no longer uses _timestamp)
            print(" - Extracting temporal features...")
            if fit:
                self.temporal_extractor = TemporalFeatureExtractor(
                    datetime_col=self.datetime_col,
                    reference_date=self.reference_date #type:ignore
                )
                X = self.temporal_extractor.fit_transform(X)
            else:
                if self.temporal_extractor is None:
                    raise ValueError("Pipeline must be fitted before transform")
                X = self.temporal_extractor.transform(X)

            # 2. Extract text features
            print(" - Extracting text features...")
            text_extractor = TextFeatureExtractor(text_columns=self.text_columns)
            X = text_extractor.fit_transform(X)

            # Drop meta_tags_twitter:data1 after reading time extraction
            if 'meta_tags_twitter:data1' in X.columns: #type:ignore
                X = X.drop(columns=['meta_tags_twitter:data1']) #type:ignore

            # 3. Extract HTML features
            print(" - Extracting HTML features...")
            html_extractor = HTMLFeatureExtractor(html_columns=self.html_columns)
            X = html_extractor.fit_transform(X)

            # 4. Strip HTML tags
            print(" - Stripping HTML tags...")
            html_stripper = HTMLTagStripper(
                text_columns=self.html_columns,
                chunk_size=self.chunk_size,
                show_progress=self.show_progress
            )
            X = html_stripper.fit_transform(X)

            # 5. Extract NLP features
            print(" - Extracting NLP features...")
            nlp_extractor = NLPFeatureExtractor(text_columns=['content'])
            X = nlp_extractor.fit_transform(X)

            # 9. Encode categorical columns
            print(" - Encoding categorical columns...")
            if 'image_url' in X.columns: #type:ignore
                X['image_url'] = encode_zero_ones(X['image_url']) #type:ignore
            if 'link_tags_amphtml' in X.columns: #type:ignore
                X['link_tags_amphtml'] = encode_zero_ones(X['link_tags_amphtml']) #type:ignore
            if 'meta_tags_referrer' in X.columns: #type:ignore
                X['meta_tags_referrer'] = encode_referrer(X['meta_tags_referrer']) #type:ignore

            # 10. Scale numerical columns
            print(" - Scaling numerical columns...")
            if self.model_is_tree:
                print('Skipping scaling because model is tree-based.')
            else:
                # Filter columns that actually exist
                numerical_cols = [col for col in numerical_cols if col in X.columns] #type:ignore

                if fit:
                    self.std_scaler = StandardScaler()
                    X[numerical_cols] = self.std_scaler.fit_transform(X[numerical_cols])
                else:
                    if self.std_scaler is None:
                        raise ValueError("Pipeline must be fitted before transform")
                    X[numerical_cols] = self.std_scaler.transform(X[numerical_cols])


        if self.metadata_only:
            print('Skipping content tokenization and lemmatization.')
            X_final = X.drop(columns=['content', 'title']) #type:ignore
        else:
            # 6. Clean text
            print(" - Cleaning text data...")
            text_cleaner = TextCleaner(
                text_columns=['content'],
                drop_punctuation=self.remove_punct,
                drop_stopwords=self.remove_stopwords
            )
            X = text_cleaner.fit_transform(X)

            if self.content_only:
                # 4. Strip HTML tags
                print(" - Stripping HTML tags...")
                html_stripper = HTMLTagStripper(
                    text_columns=self.html_columns,
                    chunk_size=self.chunk_size,
                    show_progress=self.show_progress
                )
                X = html_stripper.fit_transform(X)

            # 7. Tokenize and lemmatize
            print(" - Tokenizing and lemmatizing...")
            tokenizer = TokenizerLemmatizer(text_columns=['content'])
            X = tokenizer.fit_transform(X)

            # 8. TF-IDF vectorization
            print(" - Vectorizing text data with TF-IDF...")
            if fit:
                self.tfidf_vectorizer = TfidfVectorizer(min_df=self.tf_idf_min_ratio)
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(X['content_lemmatized'])
                self.tfidf_feature_names_ = self.tfidf_vectorizer.get_feature_names_out()
            else:
                if self.tfidf_vectorizer is None:
                    raise ValueError("Pipeline must be fitted before transform")
                tfidf_matrix = self.tfidf_vectorizer.transform(X['content_lemmatized'])

            # Create TF-IDF DataFrame
            tfidf_df = pd.DataFrame(
                tfidf_matrix.toarray(), #type:ignore
                columns=self.tfidf_feature_names_,
                index=X.index#type:ignore
            )

            # Drop text columns that are no longer needed
            X = X.drop(columns=['content', 'title', 'content_lemmatized']) #type:ignore

            # 11. Concatenate all features
            X_final = pd.concat([X, tfidf_df], axis=1)

        # Add back target variable if it exists
        if y is not None:
            X_final = pd.concat([X_final, y], axis=1)

        return X_final
