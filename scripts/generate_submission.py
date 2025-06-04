import os

import joblib
import numpy as np
from catboost import CatBoostRegressor

from src import config, utils
from src.preprocessing import load_data, preprocess_data
from src.train_bert_model import get_bert_predictions


def generate_predictions():
    utils.set_seed(config.SEED)

    print("--- Loading and Preprocessing Test Data ---")
    train_df_raw_temp, test_df_raw = load_data(config.TRAIN_FILE, config.TEST_FILE)
    train_exp_median = train_df_raw_temp['experience_from'].median()
    del train_df_raw_temp

    test_df_processed, _ = preprocess_data(test_df_raw.copy(), is_train=False, train_experience_median=train_exp_median)

    print("\n--- Generating BERT Predictions for Test Set ---")
    bert_test_preds = get_bert_predictions(
        test_df_processed[config.TEXT_COLUMN].tolist(),
        model_path=config.BERT_MODEL_DIR,
        tokenizer_path=config.BERT_MODEL_DIR,
        batch_size=config.BERT_BATCH_SIZE_EVAL
    )
    print("BERT test predictions generated.")

    # Load TF-IDF + SVD transformers, generate features for test
    print("\n--- Generating TF-IDF + SVD Features for Test Set ---")
    tfidf_vectorizer = joblib.load(config.TFIDF_VECTORIZER_PATH)
    svd_transformer = joblib.load(config.SVD_TRANSFORMER_PATH)

    X_test_tfidf = tfidf_vectorizer.transform(test_df_processed[config.TEXT_COLUMN])
    X_test_svd = svd_transformer.transform(X_test_tfidf)
    print("TF-IDF+SVD test features generated.")

    # Load tabular preprocessor, transform test data
    print("\n--- Preprocessing Tabular Features for Test Set ---")
    tabular_preprocessor = joblib.load(config.TABULAR_PREPROCESSOR_PATH)
    X_test_tabular_processed = tabular_preprocessor.transform(
        test_df_processed[config.NUMERICAL_FEATURES + config.CATEGORICAL_FEATURES])

    # Load trained CatBoost, LGBM and Ridge models, predict on test
    X_test_full = np.hstack([X_test_tabular_processed, X_test_svd, bert_test_preds.reshape(-1, 1)])

    print("\n--- Generating Predictions from Tabular Models (Test Set) ---")
    cb_model = CatBoostRegressor()
    cb_model.load_model(config.CATBOOST_MODEL_PATH)
    cb_test_preds = cb_model.predict(X_test_full)
    print("CatBoost test predictions generated.")

    lgbm_model = joblib.load(config.LGBM_MODEL_PATH)
    lgbm_test_preds = lgbm_model.predict(X_test_full)
    print("LightGBM test predictions generated.")

    ridge_model_l1 = joblib.load(config.RIDGE_FEATURES_MODEL_PATH)
    ridge_test_preds = ridge_model_l1.predict(X_test_full)
    print("Ridge (Level 1 features) test predictions generated.")

    # Load meta-model and make final predictions
    X_meta_test = np.column_stack(
        (bert_test_preds, cb_test_preds, lgbm_test_preds, ridge_test_preds))
    meta_model = joblib.load(config.META_MODEL_PATH)
    final_test_predictions = meta_model.predict(X_meta_test)

    # Generate Submission File
    print("\n--- Generating Submission File ---")
    submission_file_path = os.path.join(config.SUBMISSION_DIR, "final_manual_submission.csv")
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
    utils.create_submission_file(test_df_raw.index, final_test_predictions, submission_file_path)

    print("\n--- Submission Generation Finished ---")


if __name__ == "__main__":
    generate_predictions()
