import pandas as pd

from src import config


def load_data(train_path, test_path):
    """Loads train and test datasets."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    return train_df, test_df


def clean_text_series(text_series: pd.Series) -> pd.Series:
    """Cleans a series of text data."""
    text_series = text_series.str.replace(r'<[^>]+>', ' ', regex=True)
    text_series = text_series.str.replace(r'\n\n+', '\n', regex=True)
    text_series = text_series.str.replace(r'\t+', ' ', regex=True)
    text_series = text_series.str.replace(r' +', ' ', regex=True)
    return text_series.str.strip()


def preprocess_data(df: pd.DataFrame, is_train=True, train_experience_median=None):
    """Applies preprocessing steps to the DataFrame."""
    processed_df = df.copy()

    if is_train:
        current_experience_median = processed_df['experience_from'].median()
        processed_df['experience_from'].fillna(current_experience_median, inplace=True)
    else:
        processed_df['experience_from'].fillna(train_experience_median, inplace=True)

    text_cols_to_fill = ['title', 'location', 'company', 'skills', 'description']
    for col in text_cols_to_fill:
        if col in processed_df.columns:
            processed_df[col].fillna('', inplace=True)

    # clean location (take first word, usually city)
    processed_df['location'] = processed_df['location'].str.split(' ').str[0]

    processed_df[config.TEXT_COLUMN] = (
            processed_df['title'].astype(str) + ' ' +
            processed_df['company'].astype(str) + ' ' +
            processed_df['location'].astype(str) + ' ' +
            processed_df['description'].astype(str) + ' ' +
            processed_df['skills'].astype(str)
    )

    processed_df[config.TEXT_COLUMN] = clean_text_series(processed_df[config.TEXT_COLUMN])

    if is_train:
        return processed_df, current_experience_median

    return processed_df, None


if __name__ == '__main__':
    train_df_raw, test_df_raw = load_data(config.TRAIN_FILE, config.TEST_FILE)

    train_processed_df, train_exp_median = preprocess_data(train_df_raw, is_train=True)
    test_processed_df, _ = preprocess_data(test_df_raw, is_train=False, train_experience_median=train_exp_median)
    print("Preprocessing complete.")
