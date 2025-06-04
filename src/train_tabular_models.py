import joblib
import lightgbm as lgb
import numpy as np
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from src import config, utils
from src.feature_engineering import create_tabular_preprocessor
from src.preprocessing import load_data, preprocess_data


def train_catboost(X_train, y_train, X_val, y_val, cat_features_indices):
    print("Training CatBoost model...")
    model = CatBoostRegressor(
        iterations=config.CATBOOST_ITERATIONS,
        learning_rate=config.CATBOOST_LEARNING_RATE,
        depth=config.CATBOOST_DEPTH,
        l2_leaf_reg=config.CATBOOST_L2_LEAF_REG,
        cat_features=cat_features_indices,
        random_seed=config.SEED,
        task_type="GPU" if config.DEVICE == "cuda" else "CPU",
        verbose=100,
        eval_metric="R2",
        od_wait=150,
        use_best_model=True
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    model.save_model(config.CATBOOST_MODEL_PATH)
    print(f"CatBoost model saved to {config.CATBOOST_MODEL_PATH}")

    val_preds = model.predict(X_val)
    utils.compute_r2_score(y_val, val_preds)
    return model


def train_lgbm(X_train, y_train, X_val, y_val, cat_feature_names):
    print("Training LightGBM model...")
    model = lgb.LGBMRegressor(
        n_estimators=config.LGBM_N_ESTIMATORS,
        learning_rate=config.LGBM_LEARNING_RATE,
        max_depth=config.LGBM_MAX_DEPTH,
        objective=config.LGBM_OBJECTIVE,
        random_state=config.SEED,
        verbose=-1,
        device_type='gpu' if config.DEVICE == 'cuda' else 'cpu'
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              eval_metric='r2',
              callbacks=[lgb.early_stopping(100, verbose=False)])

    joblib.dump(model, config.LGBM_MODEL_PATH)
    print(f"LightGBM model saved to {config.LGBM_MODEL_PATH}")

    val_preds = model.predict(X_val)
    utils.compute_r2_score(y_val, val_preds)
    return model


def train_ridge_features(X_train, y_train, X_val, y_val):
    print("Training Ridge model on features...")
    model = Ridge(alpha=config.RIDGE_ALPHA, random_state=config.SEED)
    model.fit(X_train, y_train)
    joblib.dump(model, config.RIDGE_FEATURES_MODEL_PATH)
    print(f"Ridge (features) model saved to {config.RIDGE_FEATURES_MODEL_PATH}")

    val_preds = model.predict(X_val)
    utils.compute_r2_score(y_val, val_preds)
    return model


def train_meta_model(X_meta_train, y_train, X_meta_val, y_val):
    """Trains a meta-model (Ridge) on predictions from base models."""
    print("Training Meta-Model (Ridge)...")
    meta_model = Ridge(alpha=config.META_RIDGE_ALPHA, random_state=config.SEED)
    meta_model.fit(X_meta_train, y_train)
    joblib.dump(meta_model, config.META_MODEL_PATH)
    print(f"Meta-model saved to {config.META_MODEL_PATH}")

    val_preds = meta_model.predict(X_meta_val)
    utils.compute_r2_score(y_val, val_preds)
    return meta_model


if __name__ == '__main__':
    utils.set_seed(config.SEED)
    print("--- Preparing data for testing tabular models ---")

    train_df_raw, _ = load_data(config.TRAIN_FILE, config.TEST_FILE)
    train_df_processed, train_exp_median = preprocess_data(train_df_raw.copy(), is_train=True)

    train_data, val_data = train_test_split(
        train_df_processed, test_size=0.2, random_state=config.SEED
    )
    y_train = train_data[config.TARGET_COLUMN].values
    y_val = val_data[config.TARGET_COLUMN].values

    bert_train_preds = np.load(config.BERT_TRAIN_PREDS_PATH)
    bert_val_preds = np.load(config.BERT_VAL_PREDS_PATH)
    print("Loaded BERT predictions.")

    X_train_svd = np.load(config.TFIDF_TRAIN_PATH)
    X_val_svd = np.load(config.TFIDF_VAL_PATH)
    print("Loaded TF-IDF+SVD features.")

    tabular_preprocessor = create_tabular_preprocessor(
        train_data, config.NUMERICAL_FEATURES, config.CATEGORICAL_FEATURES, is_train=True
    )
    X_train_tabular_processed = tabular_preprocessor.transform(
        train_data[config.NUMERICAL_FEATURES + config.CATEGORICAL_FEATURES])
    X_val_tabular_processed = tabular_preprocessor.transform(
        val_data[config.NUMERICAL_FEATURES + config.CATEGORICAL_FEATURES])

    X_train_full = np.hstack([
        X_train_tabular_processed,
        X_train_svd,
        bert_train_preds.reshape(-1, 1)
    ])
    X_val_full = np.hstack([
        X_val_tabular_processed,
        X_val_svd,
        bert_val_preds.reshape(-1, 1)
    ])
    print(f"Shape of X_train_full: {X_train_full.shape}, X_val_full: {X_val_full.shape}")

    ohe_feature_names = tabular_preprocessor.named_transformers_['ohe'].get_feature_names_out(
        config.CATEGORICAL_FEATURES)
    num_ohe_features = len(ohe_feature_names)

    cat_features_indices_for_boosters = list(range(num_ohe_features))

    print("\n--- Training CatBoost (Level 1) ---")
    catboost_model = train_catboost(X_train_full, y_train, X_val_full, y_val, cat_features_indices_for_boosters)
    cb_val_preds = catboost_model.predict(X_val_full)
    cb_train_preds_for_meta = catboost_model.predict(X_train_full)

    print("\n--- Training LightGBM (Level 1) ---")
    lgbm_model = train_lgbm(X_train_full, y_train, X_val_full, y_val, cat_features_indices_for_boosters)
    lgbm_val_preds = lgbm_model.predict(X_val_full)
    lgbm_train_preds_for_meta = lgbm_model.predict(X_train_full)

    print("\n--- Training Ridge on Features (Level 1) ---")
    ridge_model_l1 = train_ridge_features(X_train_full, y_train, X_val_full, y_val)
    ridge_val_preds = ridge_model_l1.predict(X_val_full)
    ridge_train_preds_for_meta = ridge_model_l1.predict(X_train_full)

    X_meta_train = np.column_stack(
        (bert_train_preds, cb_train_preds_for_meta, lgbm_train_preds_for_meta, ridge_train_preds_for_meta))
    X_meta_val = np.column_stack((bert_val_preds, cb_val_preds, lgbm_val_preds, ridge_val_preds))

    print("\n--- Training Meta-Model (Level 2) ---")
    meta_model_trained = train_meta_model(X_meta_train, y_train, X_meta_val, y_val)

    print("\n--- Tabular models training and meta-modeling test finished ---")
