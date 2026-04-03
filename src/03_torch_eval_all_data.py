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
from sklearn.metrics import f1_score
import json

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

# Ignore warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

TASK_TYPE = "date" # specify the task type ("topic" or "date")
MODEL = "BERT" # specify the model type ("BERT", "RoBERTa", "DeBERTa")
RESULTS_DIR_PATH = f"results_all_data/results_{MODEL}_{TASK_TYPE}"
os.makedirs(RESULTS_DIR_PATH, exist_ok=True)



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
    Builds a BERT model (bert-base-cased) for sequence classification.

    Args:
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for AdamW optimizer.
        epochs (int): Number of training epochs.

    Returns:
        model (BertForSequenceClassification): HuggingFace BERT model.
        train_args (TrainingArguments): Default training configuration.
    """
    
    if MODEL == "BERT":
        model = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
    elif MODEL == "DeBERTa":
        model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-base", num_labels=2)
    elif MODEL == "RoBERTa":
        model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
    else:
        raise ValueError("Invalid MODEL. Choose either 'BERT', 'DeBERTa', or 'RoBERTa'.")

    train_args = TrainingArguments(
        output_dir="./tmp",
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
    dataset_df = data_by_topic()
elif TASK_TYPE == "date":
    dataset_df = data_by_date()
else:
    raise ValueError("Invalid TASK_TYPE. Choose either 'topic' or 'date'.")

print(f"--- DATASET SIZES BY {TASK_TYPE.upper()} ---")
for task, df in dataset_df.items():
    print(f"{TASK_TYPE.capitalize()}: {task}, Number of samples: {len(df)}")

# split all datasets in train/val/test
datasets = {task: split_dataset(df) for task, df in dataset_df.items()} 

if MODEL == "BERT":
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
elif MODEL == "DeBERTa":
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
elif MODEL == "RoBERTa":
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
else:
    raise ValueError("Invalid MODEL. Choose either 'BERT', 'DeBERTa', or 'RoBERTa'.")

print("\nComputing tokenized datasets...")
datasets =  tokenize_datasets(datasets, tokenizer) # tokenize all datasets



# ------------------------------------
# Fine-tuning on Dataset by Topic/Date
# ------------------------------------

path = f"{RESULTS_DIR_PATH}/best_params"
with open(f"{path}.json", "r") as f:
    best_params = json.load(f)

best_model, train_args = build_model(
    learning_rate=best_params["learning_rate"],
    weight_decay=best_params["weight_decay"],
    epochs=3
)


# merge all training and validation data across topics/dates for optimization
X_train_list = [datasets[task]['train'][0] for task in datasets]
X_val_list = [datasets[task]['val'][0] for task in datasets]
train_dataset = concatenate_datasets(X_train_list)
val_dataset = concatenate_datasets(X_val_list)
# shuffle the combined training and validation data
train_dataset = train_dataset.shuffle(seed=42)
val_dataset = val_dataset.shuffle(seed=42)

# test set for balanced full evaluation (via sample weights)
X_full_test = np.concatenate([datasets[t]["test"][0] for t in datasets])
y_full_test = np.concatenate([datasets[t]["test"][1] for t in datasets])
# sample weights for full evaluation
sample_weights = []
sample_weights_full = []
for t in datasets:
    X_values, y_values = datasets[t]["test"]
    n = len(y_values)
    w = 1.0 / n  # weight for each sample in this topic/date
    sample_weights.append(np.full(n, w))
sample_weights_full = np.concatenate(sample_weights)

# training
print(f"\n=== Training/Fine-tuning on {TASK_TYPE} ===")


"""# Update training arguments for evaluation strategy
train_args.eval_strategy = "epoch"     # Evaluate at end of each epoch
train_args.save_strategy = "epoch"     # Save at end of each epoch
train_args.load_best_model_at_end = True # Load best model based on validation metric
train_args.metric_for_best_model = "eval_loss"
train_args.greater_is_better = False"""

# define trainer
trainer = Trainer(
    model=best_model,
    args=train_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# fine-tune on train + val
trainer.train()


# evaluation on all topics/dates
print(f"\n--- Evaluation on all {TASK_TYPE}s ---")
results_task = {}
for task, data in datasets.items(): # for each topic/date
    X_te, y_te = data["test"]
    preds = trainer.predict(X_te)
    preds = np.argmax(preds.predictions, axis=1)
    f1 = f1_score(y_te, preds, average="weighted")
    results_task[task] = f1
    print(f"{TASK_TYPE.capitalize()}: {task}, Weighted F1-score: {results_task[task]:.4f}")

# evaluation on full test set
print("\n--- Evaluation on full test set ---")
preds_full = trainer.predict(X_full_test)
preds_full = np.argmax(preds_full.predictions, axis=1)
f1_full = f1_score(y_full_test, preds_full, average="weighted", sample_weight=sample_weights_full)
results_full = f1_full
print(f"Full Test Set Weighted F1-score: {results_full:.4f}")



# save results for Single Topic/Date
path = f"{RESULTS_DIR_PATH}/results_single_{TASK_TYPE}.json"
with open(path, "w") as f:
    json.dump(results_task, f, indent=4)

# save results for Full Test Set
path = f"{RESULTS_DIR_PATH}/results_full_{TASK_TYPE}.json"
with open(path, "w") as f:
    json.dump(results_full, f, indent=4)