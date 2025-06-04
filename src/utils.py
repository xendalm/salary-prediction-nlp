import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score


def set_seed(seed: int = 42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_r2_score(y_true, y_pred):
    """Computes and prints the R2 score."""
    score = r2_score(y_true, y_pred)
    print(f"R2 Score: {score:.6f}")
    return score


def create_submission_file(test_ids, predictions, file_path, id_col='index', pred_col='prediction'):
    """Creates a submission CSV file."""
    submission_df = pd.DataFrame({id_col: test_ids, pred_col: predictions})
    submission_df.to_csv(file_path, index=False)
    print(f"Submission file created at: {file_path}")
