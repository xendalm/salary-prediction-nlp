import os

import numpy as np
from sklearn.model_selection import train_test_split

from src import config, utils
from src.feature_engineering import create_tfidf_svd_features, create_tabular_preprocessor
from src.preprocessing import load_data, preprocess_data
from src.train_bert_model import fine_tune_bert_with_trainer, get_bert_predictions
from src.train_tabular_models import train_catboost, train_lgbm, train_meta_model


def main():
    utils.set_seed(config.SEED)

    # 1. Load and preprocess data
    print("--- Loading and Preprocessing Data ---")
    train_df_raw, test_df_raw = load_data(config.TRAIN_FILE, config.TEST_FILE)
    train_df_processed, train_exp_median = preprocess_data(train_df_raw.copy(), is_train=True)
    test_df_processed, _ = preprocess_data(test_df_raw.copy(), is_train=False, train_experience_median=train_exp_median)

    train_data, val_data = train_test_split(
        train_df_processed, test_size=0.2, random_state=config.SEED
    )
    y_train = train_data[config.TARGET_COLUMN]
    y_val = val_data[config.TARGET_COLUMN]

    # 2. BERT Fine-tuning with Trainer
    print("\n--- BERT Fine-tuning with Custom BertRegressor ---")
    fine_tune_bert_with_trainer(train_data, val_data, config.TEXT_COLUMN, config.TARGET_COLUMN)

    print("Generating BERT predictions for train, val, and test sets...")
    bert_train_preds = get_bert_predictions(
        train_data[config.TEXT_COLUMN].tolist(),
        model_path=config.BERT_MODEL_DIR,
        tokenizer_path=config.BERT_MODEL_DIR,
        batch_size=config.BERT_BATCH_SIZE_EVAL
    )
    np.save(config.BERT_TRAIN_PREDS_PATH, bert_train_preds)

    bert_val_preds = get_bert_predictions(
        val_data[config.TEXT_COLUMN].tolist(),
        model_path=config.BERT_MODEL_DIR,
        tokenizer_path=config.BERT_MODEL_DIR,
        batch_size=config.BERT_BATCH_SIZE_EVAL
    )
    np.save(config.BERT_VAL_PREDS_PATH, bert_val_preds)

    bert_test_preds = get_bert_predictions(
        test_df_processed[config.TEXT_COLUMN].tolist(),
        model_path=config.BERT_MODEL_DIR,
        tokenizer_path=config.BERT_MODEL_DIR,
        batch_size=config.BERT_BATCH_SIZE_EVAL
    )
    np.save(config.BERT_TEST_PREDS_PATH, bert_test_preds)
    print("BERT predictions saved.")

    # 3. TF-IDF + SVD Feature Engineering
    print("\n--- TF-IDF + SVD Feature Engineering ---")
    X_train_svd, X_val_svd_from_train = create_tfidf_svd_features(train_data[config.TEXT_COLUMN],
                                                                  val_data[config.TEXT_COLUMN], is_train=True)
    _, X_test_svd = create_tfidf_svd_features(None, test_df_processed[config.TEXT_COLUMN], is_train=False)

    np.save(config.TFIDF_TRAIN_PATH, X_train_svd)
    np.save(config.TFIDF_VAL_PATH, X_val_svd_from_train)
    np.save(config.TFIDF_TEST_PATH, X_test_svd)
    print("TF-IDF+SVD features saved.")

    # 4. Tabular Feature Preprocessing
    print("\n--- Tabular Feature Preprocessing ---")
    tabular_preprocessor = create_tabular_preprocessor(
        train_data, config.NUMERICAL_FEATURES, config.CATEGORICAL_FEATURES, is_train=True
    )
    X_train_tabular_processed = tabular_preprocessor.transform(
        train_data[config.NUMERICAL_FEATURES + config.CATEGORICAL_FEATURES])
    X_val_tabular_processed = tabular_preprocessor.transform(
        val_data[config.NUMERICAL_FEATURES + config.CATEGORICAL_FEATURES])
    X_test_tabular_processed = tabular_preprocessor.transform(
        test_df_processed[config.NUMERICAL_FEATURES + config.CATEGORICAL_FEATURES])

    # 5. Combine features for tabular models
    print("\n--- Combining features for tabular models ---")
    X_train_full = np.hstack([X_train_tabular_processed, X_train_svd, bert_train_preds.reshape(-1, 1)])
    X_val_full = np.hstack([X_val_tabular_processed, X_val_svd_from_train, bert_val_preds.reshape(-1, 1)])
    X_test_full = np.hstack([X_test_tabular_processed, X_test_svd, bert_test_preds.reshape(-1, 1)])

    ohe_feature_names = tabular_preprocessor.named_transformers_['ohe'].get_feature_names_out(
        config.CATEGORICAL_FEATURES)
    cat_features_indices_for_catboost = list(range(len(ohe_feature_names)))

    # 6. Train Tabular Models
    print("\n--- Training CatBoost ---")
    cb_model = train_catboost(X_train_full, y_train, X_val_full, y_val, cat_features_indices_for_catboost)
    cb_train_preds = cb_model.predict(X_train_full)
    cb_val_preds = cb_model.predict(X_val_full)
    cb_test_preds = cb_model.predict(X_test_full)

    print("\n--- Training LightGBM ---")
    lgbm_model = train_lgbm(X_train_full, y_train, X_val_full, y_val, cat_features_indices_for_catboost)
    lgbm_train_preds = lgbm_model.predict(X_train_full)
    lgbm_val_preds = lgbm_model.predict(X_val_full)
    lgbm_test_preds = lgbm_model.predict(X_test_full)

    # 7. Meta-Modeling
    print("\n--- Training Meta-Model ---")
    X_meta_train = np.column_stack((bert_train_preds, cb_train_preds, lgbm_train_preds))
    X_meta_val = np.column_stack((bert_val_preds, cb_val_preds, lgbm_val_preds))
    X_meta_test = np.column_stack((bert_test_preds, cb_test_preds, lgbm_test_preds))

    meta_model = train_meta_model(X_meta_train, y_train, X_meta_val, y_val)
    final_test_predictions = meta_model.predict(X_meta_test)

    # 8. Generate Submission File
    print("\n--- Generating Submission File ---")
    submission_file_path = os.path.join(config.SUBMISSION_DIR, "final_manual_submission.csv")
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
    utils.create_submission_file(test_df_raw.index, final_test_predictions, submission_file_path)

    print("\n--- Full Manual Pipeline Finished ---")


if __name__ == "__main__":
    main()
