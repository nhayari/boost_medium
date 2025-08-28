import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from typing import List
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from html import unescape
from textstat import flesch_reading_ease, flesch_kincaid_grade
from utils.text_preprocessing import remove_non_ascii, remove_punctuation, remove_stopwords, replace_numbers, remove_extra_whitespace
from medium.ml_logic.data import clean_data
# from sklearn.pipeline import make_pipeline
# from sklearn.compose import ColumnTransformer, make_column_transformer
# from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

HTML_TAG_PATTERN = re.compile(r'<[^>]+>')
WHITESPACE_PATTERN = re.compile(r'\s+')

def extract_temporal_features(df: pd.DataFrame, datetime_col: str, drop: bool = False) -> pd.DataFrame:
    """
    Extract temporal features from a datetime column in the dataframe.
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing the datetime column.
    datetime_col : str
        Name of the datetime column to extract features from.
    drop : bool
        Whether to drop the original datetime column after feature extraction. Default is False.
    Returns:
    --------
    pandas.DataFrame
        Dataframe with new temporal features added.
    """

    df_temp = df.copy()
    df_temp[datetime_col] = pd.to_datetime(df_temp[datetime_col])
    df_temp['publication_year'] = df_temp[datetime_col].dt.year
    df_temp['publication_month'] = df_temp[datetime_col].dt.month
    df_temp['publication_day'] = df_temp[datetime_col].dt.day
    df_temp['publication_dayofweek'] = df_temp[datetime_col].dt.day_of_week
    df_temp['publication_hour'] = df_temp[datetime_col].dt.hour
    df_temp['publication_is_weekend'] = df_temp['publication_dayofweek'].isin([5, 6]).astype(int)
    df_temp['days_since_publication'] = (pd.to_datetime(df_temp['_timestamp'], unit='s', utc=True) - df_temp[datetime_col]).dt.days

    if drop:
        df_temp = df_temp.drop(columns=[datetime_col])

    return df_temp

def extract_text_features(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """
    Extract basic text features from a text column in the dataframe.
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing the text column.
    text_col : str
        Name of the text column to extract features from.
    Returns:
    --------
    pandas.DataFrame
        Dataframe with new text features added.
    """

    df_text = df.copy()
    df_text[f'{text_col}_length'] = df_text[text_col].astype(str).apply(len)
    df_text[f'{text_col}_word_count'] = df_text[text_col].astype(str).apply(lambda x: len(x.split()))
    df_text[f'{text_col}_unique_word_count'] = df_text[text_col].astype(str).apply(lambda x: len(set(x.split())))
    df_text[f'{text_col}_has_numbers'] = df_text[text_col].astype(str).apply(lambda x: int(any(char.isdigit() for char in x)))
    df_text[f'{text_col}_is_question'] = df_text[text_col].astype(str).apply(lambda x: int(x.strip().endswith('?')))

    # Reading time
    if 'reading_time' not in df_text.columns:
        df_text['reading_time'] = df_text['meta_tags_twitter:data1'].astype(str).str.extract(r'(\d+)').astype(float)

    return df_text

def extract_html_features(df: pd.DataFrame, html_col: str) -> pd.DataFrame:
    """
    Extract basic HTML features from a HTML content column in the dataframe.
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing the HTML content column.
    html_col : str
        Name of the HTML content column to extract features from.
    Returns:
    --------
    pandas.DataFrame
        Dataframe with new HTML features added.
    """

    df_html = df.copy()
    df_html[f'{html_col}_num_links'] = df_html[html_col].astype(str).apply(lambda x: len(re.findall(r'http[s]?://', x)))
    df_html[f'{html_col}_num_images'] = df_html[html_col].astype(str).apply(lambda x: len(re.findall(r'<img ', x)))
    df_html[f'{html_col}_num_lists'] = df_html[html_col].astype(str).apply(lambda x: len(re.findall(r'<ul|<ol', x)))
    df_html[f'{html_col}_num_paragraphs'] = df_html[html_col].astype(str).apply(lambda x: len(re.findall(r'<p', x)))
    df_html[f'{html_col}_num_h1'] = df_html[html_col].astype(str).apply(lambda x: len(re.findall(r'<h1', x)))
    df_html[f'{html_col}_num_h2'] = df_html[html_col].astype(str).apply(lambda x: len(re.findall(r'<h2', x)))
    df_html[f'{html_col}_num_h3'] = df_html[html_col].astype(str).apply(lambda x: len(re.findall(r'<h3', x)))

    return df_html

def strip_html_tags(series: pd.Series, chunk_size: int, show_progress: bool = True) -> pd.Series:
    """
    Memory-efficient processing with optional progress tracking.
    Processes data in-place to minimize memory usage.
    """

    def strip_tags_regex_compiled(html):
        """
        Use pre-compiled regex patterns for maximum speed.
        """
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

        if show_progress:
            progress_bar.update(end_idx - i)

    if show_progress:
        progress_bar.close()

    return series

def extract_nlp_features(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """
    Extract NLP features from a text column in the dataframe.
    Metrics include readability scores and sentiment analysis.
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing the text column.
    text_col : str
        Name of the text column to extract features from.
    Returns:
    --------
    pandas.DataFrame
        Dataframe with new NLP features added."""
    # Readability scores
    df['readability_score'] = df[f'{text_col}'].apply(flesch_reading_ease)
    df['grade_level'] = df[f'{text_col}'].apply(flesch_kincaid_grade)

    # Sentiment analysis
    # Polarity score between -1 (negative) and 1 (positive)
    df['title_sentiment'] = df['title'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['content_sentiment'] = df[f'{text_col}'].apply(
        lambda x: TextBlob(x[:5000]).sentiment.polarity if len(x) > 0 else 0
    )

    return df

def clean_text(df: pd.DataFrame, text_col: str, drop_punctuation: bool,
               drop_stopwords: bool) -> pd.DataFrame:
    """
    Clean text data by lowercasing, removing punctuation, numbers, and extra whitespace.
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing the text column.
    text_col : str
        Name of the text column to clean.
    Returns:
    --------
    pandas.DataFrame
        Dataframe with cleaned text column.
    """

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
        text = remove_extra_whitespace(text)  # Remove extra whitespace
        return text

    df_cleaned = df.copy()
    df_cleaned[text_col] = df_cleaned[text_col].apply(clean_single_text)

    return df_cleaned

def tokenize_and_lemmatize(text):
    #tokenize
    tokens = word_tokenize(text)
    tokens = replace_numbers(tokens)  # Remove punctuation/numbers

    #lemmatize
    tokens_lemmatized = [WordNetLemmatizer().lemmatize(word)
                         for word in tokens
                         ]

    return (' '.join(tokens_lemmatized)).strip()

def preprocess_features(df: pd.DataFrame, chunksize: int, remove_punct: bool,
                        remove_stopwords: bool) -> np.ndarray:
    print("🎬 Preprocessing start... \n")
    df_processing = df.copy()
    df_processing = clean_data(df_processing)
    print(df_processing.shape)

    print(" - Removing unnecessary columns...")
    cols_to_remove = [
        '_id', 'tags', 'link_tags_author', 'link_tags_alternate', 'link_tags_stylesheet',
        'link_tags_apple-touch-icon', 'meta_tags_twitter:app:url:iphone', 'meta_tags_al:ios:url',
        'meta_tags_al:android:url', 'meta_tags_al:web:url', 'meta_tags_og:title', 'meta_tags_og:description',
        'meta_tags_og:url', 'meta_tags_og:description', 'meta_tags_twitter:description', 'meta_tags_author',
        'meta_tags_twitter:card', 'meta_tags_article:publisher', 'meta_tags_article:author', 'meta_tags_article:published_time',
        'meta_tags_twitter:creator', 'meta_tags_twitter:site', 'meta_tags_og:site_name', 'meta_tags_og:image', 'meta_tags_twitter:image:src',
        'meta_tags_title', 'link_tags_canonical', 'meta_tags_description', 'author_url', 'author_twitter'
    ]

    df_processing = df_processing.drop(columns=[col for col in cols_to_remove if col in df_processing.columns])

    print(" - Extracting temporal features...")
    df_processing_temporal_features = extract_temporal_features(df_processing, datetime_col='published_$date', drop=True)
    df_processing_temporal_features = df_processing_temporal_features.drop(columns='_timestamp')

    print(" - Extracting text features...")
    df_processing_title_features = extract_text_features(df_processing_temporal_features, text_col='title')
    df_processing_content_features = extract_text_features(df_processing_title_features, text_col='content')
    df_processing_content_features = df_processing_content_features.drop(columns='meta_tags_twitter:data1')

    print(" - Extracting HTML features...")
    df_processing_html_features = extract_html_features(df_processing_content_features, html_col='content')

    print(" - Stripping HTML tags...")
    df_processing_stripped_html = df_processing_html_features.copy()
    df_processing_stripped_html['content'] = strip_html_tags(df_processing_stripped_html['content'], chunk_size=chunksize, show_progress=True)

    print(" - Extracting NLP features...")
    df_processing_nlp = extract_nlp_features(df_processing_stripped_html, text_col='content')

    print(" - Cleaning text data...")
    df_processing_cleaned = clean_text(df_processing_nlp, text_col='content', drop_punctuation=remove_punct, drop_stopwords=remove_stopwords)
    df_final = df_processing_cleaned.copy()

    # Instantiating the TfidfVectorizer
    print(" - Vectorizing text data with TF-IDF...")
    tf_idf_vectorizer = TfidfVectorizer(min_df=0.2)
    df_content = df_final[['content']]

    y = df_final['log1p_recommends']

    df_final['text_lemmatized'] = df_content['content'].apply(tokenize_and_lemmatize)
    df_final = df_final.drop(columns=['content', 'title'])

    # Training it on the texts
    X_processed = tf_idf_vectorizer.fit_transform(df_final['text_lemmatized'])
    df_final = df_final.drop(columns=['text_lemmatized'])
    X_processed = pd.DataFrame(X_processed.toarray(),
                    columns = tf_idf_vectorizer.get_feature_names_out(),
                    index=df_final.index)
    print(X_processed.shape)

    # Concat
    print(df_final.shape, X_processed.shape, y.shape)
    df_processed_final = pd.concat([df_final, X_processed, y], axis=1)
    print(df_processed_final.shape)
    print("🏁 preprocess_features() done \n")

    return df_processed_final, tf_idf_vectorizer
