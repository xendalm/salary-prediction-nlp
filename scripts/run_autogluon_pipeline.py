import os

from src import config, utils
from src.preprocessing import load_data, preprocess_data
from src.train_autogluon import prepare_autogluon_data, run_autogluon_training


def main():
    utils.set_seed(config.SEED)

    print("--- Loading and Preprocessing Base Data for AutoGluon ---")
    train_df_raw, test_df_raw = load_data(config.TRAIN_FILE, config.TEST_FILE)
    train_df_processed, train_exp_median = preprocess_data(train_df_raw.copy(), is_train=True)
    test_df_processed, _ = preprocess_data(test_df_raw.copy(), is_train=False, train_experience_median=train_exp_median)

    print("\n--- Preparing Data Specifically for AutoGluon ---")
    ag_train_data, ag_val_data, ag_test_data = prepare_autogluon_data(
        train_df_processed,
        test_df_processed,
        use_bert_preds=True,
        use_tfidf_svd=True
    )

    print("\n--- Running AutoGluon Training ---")
    _, ag_test_predictions = run_autogluon_training(ag_train_data, ag_test_data, tuning_data=ag_val_data)

    print("\n--- Generating AutoGluon Submission File ---")
    submission_file_path = os.path.join(config.SUBMISSION_DIR, "final_autogluon_submission.csv")
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
    utils.create_submission_file(test_df_raw.index, ag_test_predictions, submission_file_path)

    print("\n--- AutoGluon Pipeline Finished ---")


if __name__ == "__main__":
    main()
