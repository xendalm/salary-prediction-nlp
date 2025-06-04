import os

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split

from src import config, utils


def prepare_autogluon_data(train_df_processed, test_df_processed, use_bert_preds=True, use_tfidf_svd=True):
    """Prepares data specifically for AutoGluon, including BERT and TF-IDF features if specified."""

    train_data_ag, val_data_ag = train_test_split(
        train_df_processed,
        test_size=0.2,
        random_state=config.SEED
    )

    ag_train_dfs_to_concat = [
        train_data_ag[config.CATEGORICAL_FEATURES + config.NUMERICAL_FEATURES].reset_index(drop=True),
        train_data_ag[[config.TARGET_COLUMN]].reset_index(drop=True).rename(columns={config.TARGET_COLUMN: 'target'})
    ]

    if use_bert_preds:
        print("Loading pre-computed BERT predictions for AutoGluon training set...")
        bert_train_subset_preds = np.load(config.BERT_TRAIN_PREDS_PATH)
        ag_train_dfs_to_concat.append(
            pd.DataFrame(bert_train_subset_preds, columns=['bert_pred']).reset_index(drop=True))

    if use_tfidf_svd:
        tfidf_svd_train_subset = np.load(config.TFIDF_TRAIN_PATH)
        svd_feature_names = [f'svd_{i}' for i in range(tfidf_svd_train_subset.shape[1])]
        ag_train_dfs_to_concat.append(
            pd.DataFrame(tfidf_svd_train_subset, columns=svd_feature_names).reset_index(drop=True))

    ag_train_df = pd.concat(ag_train_dfs_to_concat, axis=1)

    ag_val_dfs_to_concat = [
        val_data_ag[config.CATEGORICAL_FEATURES + config.NUMERICAL_FEATURES].reset_index(drop=True),
        val_data_ag[[config.TARGET_COLUMN]].reset_index(drop=True).rename(columns={config.TARGET_COLUMN: 'target'})
    ]
    if use_bert_preds:
        bert_val_subset_preds = np.load(config.BERT_VAL_PREDS_PATH)
        ag_val_dfs_to_concat.append(pd.DataFrame(bert_val_subset_preds, columns=['bert_pred']).reset_index(drop=True))
    if use_tfidf_svd:
        tfidf_svd_val_subset = np.load(config.TFIDF_VAL_PATH)
        ag_val_dfs_to_concat.append(
            pd.DataFrame(tfidf_svd_val_subset, columns=svd_feature_names).reset_index(drop=True))

    ag_val_df = pd.concat(ag_val_dfs_to_concat, axis=1)

    ag_full_train_df = pd.concat([ag_train_df, ag_val_df], ignore_index=True)

    ag_test_dfs_to_concat = [
        test_df_processed[config.CATEGORICAL_FEATURES + config.NUMERICAL_FEATURES].reset_index(drop=True)
    ]
    if use_bert_preds:
        bert_test_preds = np.load(config.BERT_TEST_PREDS_PATH)
        ag_test_dfs_to_concat.append(pd.DataFrame(bert_test_preds, columns=['bert_pred']).reset_index(drop=True))
    if use_tfidf_svd:
        tfidf_svd_test = np.load(config.TFIDF_TEST_PATH)
        ag_test_dfs_to_concat.append(pd.DataFrame(tfidf_svd_test, columns=svd_feature_names).reset_index(drop=True))

    ag_test_df = pd.concat(ag_test_dfs_to_concat, axis=1)

    print(f"AutoGluon training data shape: {ag_full_train_df.shape}")
    print(f"AutoGluon test data shape: {ag_test_df.shape}")

    return ag_full_train_df, ag_val_df, ag_test_df


def run_autogluon_training(train_df, test_df, tuning_data=None):
    """Trains an AutoGluon TabularPredictor."""
    utils.set_seed(config.SEED)
    os.makedirs(config.AUTOGLUON_MODEL_DIR, exist_ok=True)

    gpu_params = {'num_gpus': 1} if config.DEVICE == "cuda" else {}

    predictor = TabularPredictor(
        label='target',
        problem_type='regression',
        eval_metric=config.AUTOGLUON_EVAL_METRIC,
        path=config.AUTOGLUON_MODEL_DIR
    ).fit(
        train_data=train_df,
        tuning_data=tuning_data,
        presets=config.AUTOGLUON_PRESETS,
        time_limit=config.AUTOGLUON_TIME_LIMIT,
        ag_args_fit=gpu_params,
        # num_bag_folds=5
    )

    print("\n--- AutoGluon Leaderboard ---")
    if tuning_data is not None:
        leaderboard = predictor.leaderboard(tuning_data, silent=False)
    else:
        leaderboard = predictor.leaderboard(train_df, silent=False)
    print(leaderboard)

    print("\n--- Generating AutoGluon Predictions for Test Set ---")
    test_predictions = predictor.predict(test_df)

    return predictor, test_predictions
