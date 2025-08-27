import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
# from sklearn.pipeline import make_pipeline
# from sklearn.compose import ColumnTransformer, make_column_transformer
# from sklearn.preprocessing import OneHotEncoder, FunctionTransformer



def preprocess_features(data: pd.DataFrame) -> np.ndarray:
    print("🎬 preprocess_features starting ................\n")
    # Instantiating the TfidfVectorizer
    tf_idf_vectorizer = TfidfVectorizer(min_df=0.2)

    data['text_lemmatized'] = data['content'].apply(tokenize_and_lemmatize)

    # Training it on the texts
    X_processed = tf_idf_vectorizer.fit_transform(data['text_lemmatized'])
    X_processed = pd.DataFrame(X_processed.toarray(),
                    columns = tf_idf_vectorizer.get_feature_names_out())

    # Concat
    df_processed = pd.concat([X_processed, data['log1p_recommends']], axis=1)
    print("🏁 preprocess_features() done \n")

    return df_processed, tf_idf_vectorizer


def tokenize_and_lemmatize(text):
    #tokenize
    tokens = word_tokenize(text)

    #lemmatize
    tokens_lemmatized = [WordNetLemmatizer().lemmatize(word)
                         for word in tokens
                         ]

    return (' '.join(tokens_lemmatized)).strip()
