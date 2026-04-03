import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # "0" o "1"

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

from utils.utils_data_preprocessing import *
from utils.utils import *

import numpy as np
from datasets import Dataset
from datasets import concatenate_datasets
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import json

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# Ignore warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

RESULTS_DIR_PATH = "results/results_BERT_date"
bool_150 = False
bool_150test = False
bool_balanced_test = False
PATH_SUFFIX = create_path_suffix(
    bool_150=bool_150,
    bool_150test=bool_150test,
    bool_balanced_test=bool_balanced_test,
)



# ----------------------
# Tokenization functions
# ----------------------

def tokenize_function(examples, tokenizer, max_len=128):
    """
    Tokenizes the input examples using the provided tokenizer.

    Args:
        examples (dict): A dictionary containing the text data to be tokenized.
        tokenizer (AutoTokenizer): The tokenizer to use for tokenization.
        max_len (int): Maximum length for padding/truncation.

    Returns:
        dict: Tokenized inputs with padding and truncation applied.
    """

    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_len,
    )


def tokenize_datasets(datasets, tokenizer):
    """
    Tokenizes multiple datasets using the provided tokenizer.

    Args:
        datasets (dict): A dictionary where keys are dataset names and values are dictionaries with 'train', 'val', and 'test' splits.
        tokenizer (AutoTokenizer): The tokenizer to use for tokenization.

    Returns:
        dict: A dictionary with the same keys as input datasets, but with tokenized datasets.
    """

    tokenized_datasets = {}
    
    for task, data in datasets.items():
        print(f"\n=== Tokenizing dataset: {task} ===")

        train_dataset = Dataset.from_dict({"text": data["train"][0], "label": data["train"][1].astype(int)})
        val_dataset = Dataset.from_dict({"text": data["val"][0], "label": data["val"][1].astype(int)})
        test_dataset = Dataset.from_dict({"text": data["test"][0], "label": data["test"][1].astype(int)})

        train_tokenized = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
        val_tokenized = val_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
        test_tokenized = test_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

        tokenized_datasets[task] = {
            "train": (train_tokenized, np.array(train_dataset["label"])),
            "val": (val_tokenized, np.array(val_dataset["label"])),
            "test": (test_tokenized, np.array(test_dataset["label"]))
        }

    return tokenized_datasets



# --------------------
# Build model function
# --------------------

def build_model(learning_rate=2e-5, weight_decay=0.1, epochs=1):
    """
    Builds a BERT model (bert-base-cased) for sequence classification.

    Args:
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for AdamW optimizer.
        epochs (int): Number of training epochs.

    Returns:
        model (BertForSequenceClassification): HuggingFace BERT model.
        train_args (TrainingArguments): Default training configuration.
    """
    
    model = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

    train_args = TrainingArguments(
        output_dir="./tmp",
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        optim="adamw_torch_fused", # fast

        save_strategy="no",
        eval_strategy="no",
        logging_strategy="no",
        report_to="none",
        load_best_model_at_end=False,
    )

    return model, train_args



# ----------------
# Data preparation
# ----------------

dataset_df = data_by_date()

print("--- DATASET SIZES BY DATE ---")
for date, df in dataset_df.items():
    print(f"Date: {date}, Number of samples: {len(df)}")

#dataset_df = dict(list(dataset_df.items())[6:]) # remove firsts datasets

print("\nSplitting datasets into train/val/test...")
if bool_150: # split all datasets in train/val/test (max 150 samples in val and test)
    datasets = {date: split_dataset_150(df) for date, df in dataset_df.items()}
else: # split all datasets in train/val/test
    datasets = {date: split_dataset(df) for date, df in dataset_df.items()} 

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

print("\nComputing tokenized datasets...")
datasets =  tokenize_datasets(datasets, tokenizer) # tokenize all datasets



# ------------------------------
# Fine-tuning on Dataset by Date
# ------------------------------

path = f"{RESULTS_DIR_PATH}/best_params"
if bool_150: path += "_150" # add "_150" suffix to path, if bool_150 is True
with open(f"{path}.json", "r") as f:
    best_params = json.load(f)

best_model, train_args = build_model(
    learning_rate=best_params["learning_rate"],
    weight_decay=best_params["weight_decay"],
    epochs=3
)

results_date = {}
results_full = {}

if bool_150test: # balanced full test set (max length 150)
    X_trimmed_list = []
    y_trimmed_list = []
    for t in datasets:
        X_test, y_test = datasets[t]["test"]
        X_trim, y_trim, _, _ = trim_and_extract(X_test, y_test, max_len=150)
        X_trimmed_list.append(X_trim)
        y_trimmed_list.append(y_trim)
    X_full_test = concatenate_datasets(X_trimmed_list)
    y_full_test = np.concatenate(y_trimmed_list)
elif bool_balanced_test: # balanced full test set (via sample weights)
    X_full_test = np.concatenate([datasets[t]["test"][0] for t in datasets])
    y_full_test = np.concatenate([datasets[t]["test"][1] for t in datasets])
    sample_weights = []
    for t in datasets:
        _, y_test_t = datasets[t]["test"]
        n = len(y_test_t)
        w = 1.0 / n  # weight for each sample in this topic
        sample_weights.extend([w] * n)
    sample_weights = np.array(sample_weights)
else: # unbalanced full test set
    X_full_test = np.concatenate([datasets[t]["test"][0] for t in datasets])
    y_full_test = np.concatenate([datasets[t]["test"][1] for t in datasets])

# sequential training
for i, (date, data) in enumerate(datasets.items()):
    print(f"\n=== Phase {i+1}: Training/Fine-tuning on date: {date} ===")
    
    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    X_test, y_test = data["test"]

    # define trainer
    trainer = Trainer(
        model=best_model,
        args=train_args,
        train_dataset=X_train,
        eval_dataset=X_val,
    )
    
    # fine-tune on train + val
    trainer.train()

    # evaluate on current dataset
    y_pred = trainer.predict(X_test)
    y_pred = np.argmax(y_pred.predictions, axis=1)
    print(f"\nClassification Report after date {date}:")
    print(classification_report(y_test, y_pred))
    print(f"Confusion Matrix after date {date}:")
    print(confusion_matrix(y_test, y_pred))
    print(f"Weighted F1-score after date {date}:", f1_score(y_test, y_pred, average='weighted'))

    # evaluate on all datasets
    print("\n--- Evaluation on all datasets ---")
    results_date[date] = {}
    for test_date, test_data in datasets.items(): # for each date
        X_te, y_te = test_data["test"]
        preds = trainer.predict(X_te)
        preds = np.argmax(preds.predictions, axis=1)
        f1 = f1_score(y_te, preds, average="weighted")
        results_date[date][test_date] = f1
        print(f"Evaluation on date {test_date}: Weighted F1 = {f1:.4f}")

    # evaluation on full test set
    print("\n--- Evaluation on full test set ---")
    preds_full = trainer.predict(X_full_test)
    preds_full = np.argmax(preds_full.predictions, axis=1)
    if bool_balanced_test:
        f1_full = f1_score(y_full_test, preds_full, average="weighted", sample_weight=sample_weights)
    else:
        f1_full = f1_score(y_full_test, preds_full, average="weighted")
    results_full[date] = f1_full
    print(f"Evaluation on full test set: Weighted F1 = {f1_full:.4f}")



# ---------------
# Results summary
# ---------------

print_results_summary(results_date, results_full, type="Date")

# save results for Single Date
path = f"{RESULTS_DIR_PATH}/results_single_date{PATH_SUFFIX}.json"
with open(path, "w") as f:
    json.dump(results_date, f, indent=4)

# save results for Full Test Set
path = f"{RESULTS_DIR_PATH}/results_full_date{PATH_SUFFIX}.json"
with open(path, "w") as f:
    json.dump(results_full, f, indent=4)



# -----------------------------------
# Plot on each Date and Full Test Set
# -----------------------------------

path = f"{RESULTS_DIR_PATH}/f1_by_date{PATH_SUFFIX}.png"
plot_f1_by_task(results_date, type="Date", path=path)

path = f"{RESULTS_DIR_PATH}/f1_full_test_date{PATH_SUFFIX}.png"
plot_f1_full_test(results_full, type="Date", path=path)