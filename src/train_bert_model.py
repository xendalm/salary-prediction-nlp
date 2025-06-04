import os

import numpy as np
import torch
from datasets import Dataset
from torch import nn, HuberLoss
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments

from src import config, utils


def compute_metrics_for_trainer(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.flatten()
    labels = labels.flatten()
    r2 = utils.r2_score(labels, predictions)
    return {"r2": r2}


class BertRegressor(nn.Module):
    def __init__(self, model_name):
        super(BertRegressor, self).__init__()
        self.bert = AutoModel.from_pretrained(
            model_name,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)
        self.loss_fn = HuberLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.regressor(pooled_output).squeeze(-1)

        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}


def fine_tune_bert_with_trainer(train_df, val_df, text_column, target_column):
    utils.set_seed(config.SEED)

    # Use the custom BertRegressor instead of AutoModelForSequenceClassification
    model = BertRegressor(config.BERT_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(examples[text_column], truncation=True, padding="max_length",
                         max_length=config.BERT_MAX_LENGTH)

    train_df_hf = train_df.copy()
    val_df_hf = val_df.copy()
    train_df_hf["label"] = train_df_hf[target_column].astype(np.float32)
    val_df_hf["label"] = val_df_hf[target_column].astype(np.float32)

    train_hf_dataset = Dataset.from_pandas(train_df_hf[[text_column, "label"]])
    val_hf_dataset = Dataset.from_pandas(val_df_hf[[text_column, "label"]])

    train_tokenized_dataset = train_hf_dataset.map(tokenize_function, batched=True, remove_columns=[text_column])
    val_tokenized_dataset = val_hf_dataset.map(tokenize_function, batched=True, remove_columns=[text_column])

    training_args = TrainingArguments(
        output_dir=config.BERT_MODEL_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=200,
        learning_rate=config.BERT_LEARNING_RATE,
        per_device_train_batch_size=config.BERT_BATCH_SIZE_TRAIN,
        per_device_eval_batch_size=config.BERT_BATCH_SIZE_EVAL,
        save_total_limit=2,
        num_train_epochs=config.BERT_EPOCHS,
        weight_decay=config.BERT_WEIGHT_DECAY,
        warmup_steps=config.BERT_WARMUP_STEPS,
        lr_scheduler_type=config.BERT_LR_SCHEDULER_TYPE,
        logging_dir=os.path.join(config.BERT_MODEL_DIR, "logs"),
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        seed=config.SEED,
        report_to="tensorboard",
        load_best_model_at_end=True,
        metric_for_best_model="r2",
        greater_is_better=True,
    )

    # Use the standard Trainer instead of custom RegressionTrainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized_dataset,
        eval_dataset=val_tokenized_dataset,
        compute_metrics=compute_metrics_for_trainer,
        processing_class=tokenizer
    )

    print(f"Starting BERT fine-tuning on device: {training_args.device}")
    trainer.train()

    print("Fine-tuning finished. Saving the best model and tokenizer...")
    trainer.save_model()
    tokenizer.save_pretrained(config.BERT_MODEL_DIR)


def get_bert_predictions(texts_to_predict, model_path, tokenizer_path, batch_size=32):
    """
    Generates predictions using a fine-tuned BertRegressor model.
    """
    print(f"Loading BertRegressor model from: {model_path}")
    print(f"Loading BERT tokenizer from: {tokenizer_path}")

    # Load the custom model
    model = BertRegressor(config.BERT_MODEL_NAME)
    model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location=config.DEVICE))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    model.to(config.DEVICE)
    model.eval()

    predictions = []

    for i in range(0, len(texts_to_predict), batch_size):
        batch_texts = texts_to_predict[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True,
                           max_length=config.BERT_MAX_LENGTH)
        inputs = {k: v.to(config.DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            preds = outputs["logits"].cpu().numpy().flatten()
            predictions.extend(preds)

    return np.array(predictions)


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from src.preprocessing import load_data, preprocess_data

    train_df_raw, test_df_raw = load_data(config.TRAIN_FILE, config.TEST_FILE)
    train_df_processed, train_exp_median = preprocess_data(train_df_raw, is_train=True)
    test_df_processed, _ = preprocess_data(test_df_raw, is_train=False, train_experience_median=train_exp_median)

    train_data, val_data = train_test_split(
        train_df_processed,
        test_size=0.2,
        random_state=config.SEED
    )

    print(
        f"Training BertRegressor on {len(train_data)} samples, validating on {len(val_data)} samples.")
    fine_tune_bert_with_trainer(train_data, val_data, config.TEXT_COLUMN, config.TARGET_COLUMN)

    print("Generating predictions for validation set with the best model...")
    val_preds = get_bert_predictions(
        val_data[config.TEXT_COLUMN].tolist(),
        model_path=config.BERT_MODEL_DIR,
        tokenizer_path=config.BERT_MODEL_DIR,
        batch_size=config.BERT_BATCH_SIZE_EVAL
    )
    utils.compute_r2_score(val_data[config.TARGET_COLUMN].values, val_preds)
    np.save(config.BERT_VAL_PREDS_PATH, val_preds)
    print(f"Validation predictions saved to {config.BERT_VAL_PREDS_PATH}")

    print("Generating predictions for test set...")
    test_preds = get_bert_predictions(
        test_df_processed[config.TEXT_COLUMN].tolist(),
        model_path=config.BERT_MODEL_DIR,
        tokenizer_path=config.BERT_MODEL_DIR,
        batch_size=config.BERT_BATCH_SIZE_EVAL
    )
    np.save(config.BERT_TEST_PREDS_PATH, test_preds)
    print(f"Test predictions saved to {config.BERT_TEST_PREDS_PATH}")