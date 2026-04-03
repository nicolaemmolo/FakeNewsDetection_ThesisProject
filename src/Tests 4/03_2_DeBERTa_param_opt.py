import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # "0" o "1"

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

from utils.utils_data_preprocessing import *
from utils.utils import *

import numpy as np
from datasets import Dataset
from sklearn.metrics import f1_score
import json

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

import optuna
from functools import partial
from sklearn.utils import shuffle

# Ignore warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

TASK_TYPE = "topic" # specify the task type ("topic" or "date")
RESULTS_DIR_PATH = f"results/results_DeBERTa_{TASK_TYPE}"
bool_150 = False



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



# -------------------------
# Optuna objective function
# -------------------------

def objectiveDeBERTa(trial, X_train, y_train, X_val, y_val):
    """
    Optuna objective function for hyperparameter optimization of DeBERTa model.

    Args:
        trial (optuna.trial.Trial): Optuna trial object.
        X_train (Dataset): Tokenized training dataset.
        y_train (np.array): Training labels.
        X_val (Dataset): Tokenized validation dataset.
        y_val (np.array): Validation labels.

    Returns:
        float: Validation F1 score to be maximized.
    """

    learning_rate = trial.suggest_categorical("learning_rate", [5e-5, 3e-5, 2e-5, 1e-5])
    weight_decay = trial.suggest_categorical("weight_decay", [0.0, 0.1, 0.01, 0.001])

    model, train_args = build_model(learning_rate, weight_decay, epochs=1)
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=X_train,
        eval_dataset=X_val,
    )
    
    trainer.train()
    preds = trainer.predict(X_val)
    preds = np.argmax(preds.predictions, axis=1)
    f1 = f1_score(y_val, preds, average="weighted")
    return f1



# ----------------
# Data preparation
# ----------------

if TASK_TYPE == "topic":
    dataset_df = data_by_topic()
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



# ---------------------------
# Hyperparameter optimization
# ---------------------------

# merge all training and validation data across topics/dates for optimization
X_train = np.concatenate([datasets[task]['train'][0] for task in datasets])
y_train = np.concatenate([datasets[task]['train'][1] for task in datasets])
X_val = np.concatenate([datasets[task]['val'][0] for task in datasets])
y_val = np.concatenate([datasets[task]['val'][1] for task in datasets])
# shuffle the combined training and validation data
X_train, y_train = shuffle(X_train, y_train, random_state=42)
X_val, y_val = shuffle(X_val, y_val, random_state=42)

# functool.partial to pass data to the objective function
objective_with_data = partial(objectiveDeBERTa,
                               X_train=X_train,
                               y_train=y_train,
                               X_val=X_val,
                               y_val=y_val)

study = optuna.create_study(direction="maximize") # maximize F1-score
study.optimize(objective_with_data, n_trials=30) # 30 trials; increase for better results

print("Best parameters:", study.best_params)

path = f"{RESULTS_DIR_PATH}/best_params"
if bool_150: path += "_150"
with open(f"{path}.json", "w") as f:
    json.dump(study.best_params, f, indent=4)