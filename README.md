# Salary Prediction from Job Descriptions

This project aims to predict job salaries based from descriptions and related features, primarily using NLP techniques
and ensemble modeling.

## Dataset

[Competition](https://www.kaggle.com/competitions/mts-hse-2025/data) dataset features:

- `title`: job title
- `location`: job location
- `company`: hiring company
- `skills`: required skills
- `description`: full job description
- `experience_from`: minimum years of experience
- `log_salary_from`: target (log-transformed lower bound of salary)

## Project Structure

- `data/`: Raw and processed datasets.
- `notebooks/`: EDA and experimental model development.
- `src/`: Source code for preprocessing, feature engineering, and model training scripts.
- `models/`: Saved model artifacts.
- `scripts/`: Runnable scripts for the main manual pipeline and the AutoGluon pipeline.
- `submission/`: Generated submission files.

## Methodology

The core approach involves:

1. **Preprocessing**:
    - cleaning text data using regexps
    - imputing missing `experience_from` values with the median from the training set
    - simplifying `location` to the primary city/region
2. **Feature Engineering**:
    - concatenating textual fields (`title`, `company`, `location`, `description`, `skills`) into a single text feature
    - extracting TF-IDF features from the combined text, followed by TruncatedSVD for dimensionality reduction
3. **Modeling**:
    - **BERT**: fine-tuning a pre-trained Russian BERT model on the combined text feature for regression.
    - **Tabular Models**: Training LightGBM, CatBoost and Ridge Regression models using a combination of:
        - categorical features (`location`, `company`) - one-hot encoded for LightGBM and Ridge, handled natively by
          CatBoost
        - numerical features (`experience_from`)
        - TF-IDF + SVD features
        - predictions from the fine-tuned BERT model
    - **Meta-Modeling (Stacking)**: Using Ridge regression as a meta-model to combine predictions from the BERT,
      LightGBM, and CatBoost models.
4. **Evaluation**: R2 score.

An experimental pipeline using Auto Gluon has also been developed, which uses the same set of engineered features.