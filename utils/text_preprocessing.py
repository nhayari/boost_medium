import re
import string
from typing import List, Optional
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import inflect

# --------- Nettoyage HTML  replacing by space---------
def strip_html_content(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"<.*?>", " ", text)


# --------- Nettoyage basique ---------
def remove_non_ascii(text: str) -> str:
    return re.sub(r"[^\x00-\x7F]+", " ", text) if isinstance(text, str) else ""


def to_lowercase(text: str) -> str:
    return text.lower() if isinstance(text, str) else ""


def remove_numbers(text: str) -> str:
    return re.sub(r"\d+", "", text) if isinstance(text, str) else ""

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation)) if isinstance(text, str) else ""

def remove_extra_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip() if isinstance(text, str) else ""


# --------- Stopwords (sklearn) ---------
def remove_stopwords(text: str, language: str = "english") -> str:
    if not isinstance(text, str):
        return ""
    tokens = text.split()
    stop_words = ENGLISH_STOP_WORDS if language == "english" else set()
    return " ".join([word for word in tokens if word.lower() not in stop_words])


# --------- Tokenization simple ---------
def tokenize_text(text: str) -> List[str]:
    return text.split() if isinstance(text, str) and text.strip() else []

# --------- Stemming (optionnel à la place de lemmatizer) ---------
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

def stem_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    tokens = tokenize_text(text)
    return " ".join([stemmer.stem(token) for token in tokens])

# --------- Pipelines ---------
def basic_text_clean(text: str) -> str:
    text = to_lowercase(text)
    text = remove_numbers(text)
    text = remove_punctuation(text)
    text = remove_extra_whitespace(text)
    return text

def advanced_text_clean(text: str, remove_html: bool = True, remove_stop: bool = True,
                        stem: bool = True, language: str = "english") -> str:
    if not isinstance(text, str):
        return ""
    if remove_html:
        text = strip_html_content(text)
    text = basic_text_clean(text)
    # text = remove_non_ascii(text)
    if remove_stop:
        text = remove_stopwords(text, language)
    if stem:
        text = stem_text(text)
    text = remove_extra_whitespace(text)
    return text


def preprocess_dataframe(df: pd.DataFrame, text_columns: List[str],
                        preprocessing_level: str = "advanced") -> pd.DataFrame:
    df_processed = df.copy()
    for col in text_columns:
        if col in df_processed.columns:
            if preprocessing_level == "basic":
                df_processed[f"{col}_cleaned"] = df_processed[col].apply(basic_text_clean)
            else:
                df_processed[f"{col}_cleaned"] = df_processed[col].apply(advanced_text_clean)
    return df_processed

def create_combined_text_features(df: pd.DataFrame, columns_to_combine: List[str],
                                 new_column: str = "combined_col") -> pd.DataFrame:
    df_combined = df.copy()
    columns_to_combine = [f"{col}_cleaned" if f"{col}_cleaned" in df_combined.columns else col
                          for col in columns_to_combine if col in df_combined.columns]
    if columns_to_combine:
        df_combined[new_column] = df_combined[columns_to_combine].apply(
            lambda row: " ".join(row.dropna().astype(str)), axis=1
        )
    return df_combined

# --------- Test rapide ---------
if __name__ == "__main__":
    text = "<p>test paragraph! with h1 <h1> ! title _ @ <div></div></br> </h1></p>"
    print("Original :", text)
    print("Basic clean:", basic_text_clean(text))
    print("Advanced clean:", advanced_text_clean(text))
