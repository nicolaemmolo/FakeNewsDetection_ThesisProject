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
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# Ignore warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

TASK_TYPE = "topic" # specify the task type ("topic" or "date")
RESULTS_DIR_PATH = f"results/results_DeBERTa_{TASK_TYPE}"
bool_150 = False
bool_150test = False
bool_balanced_test = False
bool_inverted = False
PATH_SUFFIX = create_path_suffix(
    bool_150=bool_150,
    bool_150test=bool_150test,
    bool_balanced_test=bool_balanced_test,
    bool_inverted=bool_inverted
)



# ----------------------
# Tokenization functions
# ----------------------

def tokenize_function(examples, tokenizer, max_len=300):
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
    
    for name, data in datasets.items():
        print(f"\n=== Tokenizing dataset: {name} ===")

        train_dataset = Dataset.from_dict({"text": data["train"][0], "label": data["train"][1].astype(int)})
        val_dataset = Dataset.from_dict({"text": data["val"][0], "label": data["val"][1].astype(int)})
        test_dataset = Dataset.from_dict({"text": data["test"][0], "label": data["test"][1].astype(int)})

        train_tokenized = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
        val_tokenized = val_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
        test_tokenized = test_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

        tokenized_datasets[name] = {
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
    Builds a DeBERTa model (microsoft/deberta-base) for sequence classification.

    Args:
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for AdamW optimizer.
        epochs (int): Number of training epochs.

    Returns:
        model (AutoModelForSequenceClassification): HuggingFace DeBERTa model.
        train_args (TrainingArguments): Default training configuration.
    """

    model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-base", num_labels=2)

    train_args = TrainingArguments(
        output_dir="./optuna_tmp",
        num_train_epochs=epochs,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
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

if TASK_TYPE == "topic":
    dataset_df = data_by_topic(inverted=bool_inverted)
elif TASK_TYPE == "date":
    dataset_df = data_by_date()
else:
    raise ValueError("Invalid TASK_TYPE. Choose either 'topic' or 'date'.")

print(f"--- DATASET SIZES BY {TASK_TYPE.upper()} ---")
for task, df in dataset_df.items():
    print(f"{TASK_TYPE.capitalize()}: {task}, Number of samples: {len(df)}")

print("\nSplitting datasets into train/val/test...")
if bool_150: # split all datasets in train/val/test (max 150 samples in val and test)
    datasets = {task: split_dataset_150(df) for task, df in dataset_df.items()}
else: # split all datasets in train/val/test
    datasets = {task: split_dataset(df) for task, df in dataset_df.items()} 

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")

print("\nComputing tokenized datasets...")
datasets =  tokenize_datasets(datasets, tokenizer) # tokenize all datasets



# ------------------------------------
# Fine-tuning on Dataset by Topic/Date
# ------------------------------------

path = f"{RESULTS_DIR_PATH}/best_params"
if bool_150: path += "_150" # add "_150" suffix to path, if bool_150 is True
with open(f"{path}.json", "r") as f:
    best_params = json.load(f)

best_model, train_args = build_model(
    learning_rate=best_params["learning_rate"],
    weight_decay=best_params["weight_decay"],
    epochs=3
)

results_task = {}
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
        w = 1.0 / n  # weight for each sample in this topic/date
        sample_weights.extend([w] * n)
    sample_weights = np.array(sample_weights)
else: # unbalanced full test set
    X_full_test = np.concatenate([datasets[t]["test"][0] for t in datasets])
    y_full_test = np.concatenate([datasets[t]["test"][1] for t in datasets])

# sequential training
for i, (task, data) in enumerate(datasets.items()):
    print(f"\n=== Phase {i+1}: Training/Fine-tuning on {TASK_TYPE}: {task} ===")

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
    print(f"\nClassification Report after {TASK_TYPE} {task}:")
    print(classification_report(y_test, y_pred))
    print(f"Confusion Matrix after {TASK_TYPE} {task}:")
    print(confusion_matrix(y_test, y_pred))
    print(f"Weighted F1-score after {TASK_TYPE} {task}:", f1_score(y_test, y_pred, average='weighted'))

    # evaluation on all topics/dates
    print(f"\n--- Evaluation on all {TASK_TYPE}s ---")
    results_task[task] = {}
    for test_task, test_data in datasets.items(): # for each topic/date
        X_te, y_te = test_data["test"]
        preds = trainer.predict(X_te)
        preds = np.argmax(preds.predictions, axis=1)
        f1 = f1_score(y_te, preds, average="weighted")
        results_task[task][test_task] = f1
        print(f"Evaluation on {TASK_TYPE} {test_task}: Weighted F1 = {f1:.4f}")

    # evaluation on full test set
    print("\n--- Evaluation on full test set ---")
    preds_full = trainer.predict(X_full_test)
    preds_full = np.argmax(preds_full.predictions, axis=1)
    if bool_balanced_test:
        f1_full = f1_score(y_full_test, preds_full, average="weighted", sample_weight=sample_weights)
    else:
        f1_full = f1_score(y_full_test, preds_full, average="weighted")
    results_full[task] = f1_full
    print(f"Evaluation on full test set: Weighted F1 = {f1_full:.4f}")



# ---------------
# Results summary
# ---------------

print_results_summary(results_task, results_full, type=TASK_TYPE.capitalize())

# save results for Single Topic/Date
path = f"{RESULTS_DIR_PATH}/results_single_{TASK_TYPE}{PATH_SUFFIX}.json"
with open(path, "w") as f:
    json.dump(results_task, f, indent=4)

# save results for Full Test Set
path = f"{RESULTS_DIR_PATH}/results_full_{TASK_TYPE}{PATH_SUFFIX}.json"
with open(path, "w") as f:
    json.dump(results_full, f, indent=4)



# -----------------------------------------
# Plot on each Topic/Date and Full Test Set
# -----------------------------------------

path = f"{RESULTS_DIR_PATH}/f1_by_{TASK_TYPE}{PATH_SUFFIX}.png"
plot_f1_by_task(results_task, type=TASK_TYPE.capitalize(), path=path)

path = f"{RESULTS_DIR_PATH}/f1_full_test_{TASK_TYPE}{PATH_SUFFIX}.png"
plot_f1_full_test(results_full, type=TASK_TYPE.capitalize(), path=path)