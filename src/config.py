import torch

DATA_DIR = "data/"
RAW_DATA_DIR = f"{DATA_DIR}raw/"
PROCESSED_DATA_DIR = f"{DATA_DIR}processed/"
MODEL_DIR = "models/"
SUBMISSION_DIR = "submission/"

TRAIN_FILE = f"{RAW_DATA_DIR}train.csv"
TEST_FILE = f"{RAW_DATA_DIR}test.csv"

BERT_MODEL_DIR = f"{MODEL_DIR}bert/"
CATBOOST_MODEL_PATH = f"{MODEL_DIR}catboost/catboost_model.cbm"
LGBM_MODEL_PATH = f"{MODEL_DIR}lgbm/lgbm_model.txt"
META_MODEL_PATH = f"{MODEL_DIR}meta_model/meta_ridge_model.joblib"
TFIDF_VECTORIZER_PATH = f"{MODEL_DIR}feature_engineering/tfidf_vectorizer.joblib"
SVD_TRANSFORMER_PATH = f"{MODEL_DIR}feature_engineering/svd_transformer.joblib"
TABULAR_PREPROCESSOR_PATH = f"{MODEL_DIR}feature_engineering/tabular_preprocessor.joblib"

AUTOGLUON_MODEL_DIR = f"{MODEL_DIR}autogluon/"

BERT_TRAIN_PREDS_PATH = f"{PROCESSED_DATA_DIR}bert_train_preds.npy"
BERT_VAL_PREDS_PATH = f"{PROCESSED_DATA_DIR}bert_val_preds.npy"
BERT_TEST_PREDS_PATH = f"{PROCESSED_DATA_DIR}bert_test_preds.npy"

TFIDF_TRAIN_PATH = f"{PROCESSED_DATA_DIR}X_train_tfidf_svd.npy"
TFIDF_VAL_PATH = f"{PROCESSED_DATA_DIR}X_val_tfidf_svd.npy"
TFIDF_TEST_PATH = f"{PROCESSED_DATA_DIR}X_test_tfidf_svd.npy"

SEED = 42
TARGET_COLUMN = "log_salary_from"
TEXT_COLUMN = "text_feature"

BERT_MODEL_NAME = "ai-forever/ruBert-base"
BERT_MAX_LENGTH = 512
BERT_BATCH_SIZE_TRAIN = 16
BERT_BATCH_SIZE_EVAL = 32
BERT_EPOCHS = 5
BERT_LEARNING_RATE = 3.5e-5
BERT_WEIGHT_DECAY = 0.085
BERT_WARMUP_STEPS = 1500
BERT_LR_SCHEDULER_TYPE = "polynomial"
BERT_MAX_GRAD_NORM = 1.0

TFIDF_MAX_FEATURES = 20000
SVD_N_COMPONENTS = 400

CATBOOST_ITERATIONS = 800
CATBOOST_LEARNING_RATE = 0.16
CATBOOST_DEPTH = 9
CATBOOST_L2_LEAF_REG = 0.00035

LGBM_N_ESTIMATORS = 550
LGBM_LEARNING_RATE = 0.13
LGBM_MAX_DEPTH = 7
LGBM_OBJECTIVE = "huber"

RIDGE_ALPHA = 1.0
RIDGE_FEATURES_MODEL_PATH = f"{MODEL_DIR}ridge_features_model.joblib"
META_RIDGE_ALPHA = 0.8

AUTOGLUON_TIME_LIMIT = 3600
AUTOGLUON_PRESETS = 'best_quality'
AUTOGLUON_EVAL_METRIC = 'r2'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CATEGORICAL_FEATURES = ['location', 'company']
NUMERICAL_FEATURES = ['experience_from']
TEXT_FEATURES_FOR_BERT = ['text_feature']
FEATURES_TO_DROP_FOR_TABULAR = ['salary_from', 'title', 'description', 'skills', TEXT_COLUMN]
