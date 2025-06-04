import joblib
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src import config


def get_stopwords_russian():
    try:
        return stopwords.words("russian")
    except LookupError:
        import nltk
        nltk.download('stopwords')
        return stopwords.words("russian")


def create_tfidf_svd_features(text_series_train: pd.Series, text_series_test: pd.Series = None, is_train=True):
    """Creates TF-IDF features followed by TruncatedSVD."""
    russian_stopwords = get_stopwords_russian()

    if is_train:
        print("Fitting TF-IDF Vectorizer and SVD Transformer...")
        vectorizer = TfidfVectorizer(
            max_features=config.TFIDF_MAX_FEATURES,
            stop_words=russian_stopwords,
            sublinear_tf=True,
            analyzer="word",
            token_pattern=r"\w{1,}",
            ngram_range=(1, 2)
        )
        X_tfidf_train = vectorizer.fit_transform(text_series_train)

        svd = TruncatedSVD(n_components=config.SVD_N_COMPONENTS, random_state=config.SEED)
        X_svd_train = svd.fit_transform(X_tfidf_train)

        joblib.dump(vectorizer, config.TFIDF_VECTORIZER_PATH)
        joblib.dump(svd, config.SVD_TRANSFORMER_PATH)
        print(f"TF-IDF Vectorizer saved to {config.TFIDF_VECTORIZER_PATH}")
        print(f"SVD Transformer saved to {config.SVD_TRANSFORMER_PATH}")

        if text_series_test is not None:
            X_tfidf_test = vectorizer.transform(text_series_test)
            X_svd_test = svd.transform(X_tfidf_test)
            return X_svd_train, X_svd_test
        return X_svd_train, None
    else:
        print("Loading TF-IDF Vectorizer and SVD Transformer...")
        vectorizer = joblib.load(config.TFIDF_VECTORIZER_PATH)
        svd = joblib.load(config.SVD_TRANSFORMER_PATH)

        X_tfidf_test = vectorizer.transform(text_series_test)
        X_svd_test = svd.transform(X_tfidf_test)
        return None, X_svd_test


def create_tabular_preprocessor(df_train: pd.DataFrame, num_features: list, cat_features: list, is_train=True):
    """Creates or loads a preprocessor for tabular (numerical and categorical) features."""
    if is_train:
        preprocessor = ColumnTransformer(
            transformers=[
                ('scaler', StandardScaler(), num_features),
                ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
            ],
            remainder='drop'
        )
        preprocessor.fit(df_train[num_features + cat_features])
        joblib.dump(preprocessor, config.TABULAR_PREPROCESSOR_PATH)
        print(f"Tabular preprocessor saved to {config.TABULAR_PREPROCESSOR_PATH}")
    else:
        print("Loading tabular preprocessor...")
        preprocessor = joblib.load(config.TABULAR_PREPROCESSOR_PATH)
    return preprocessor