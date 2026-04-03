import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # "0" o "1"

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

from utils.utils_data_preprocessing import *
from utils.utils import *
from utils.utils_continual.utils_replay import ReplayBuffer

import numpy as np
from datasets import Dataset
from datasets import concatenate_datasets
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import json

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# Ignore warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

TASK_TYPE = "topic" # specify the task type ("topic" or "date")
MODEL = "BERT" # specify the model type ("BERT", "RoBERTa", "DeBERTa")
DIM = 50 # number of samples per task to store in replay buffer
RESULTS_DIR_PATH = f"results_replay/results_{MODEL}_{TASK_TYPE}"
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

results_task = {}
results_cumulative = {}
results_full = {}

# test set for balanced cumulative/full evaluation (via sample weights)
X_cumulative_test = []
y_cumulative_test = []
X_full_test = np.concatenate([datasets[t]["test"][0] for t in datasets])
y_full_test = np.concatenate([datasets[t]["test"][1] for t in datasets])
# sample weights for cumulative/full evaluation
sample_weights = []
sample_weights_cumulative = []
sample_weights_full = []
for t in datasets:
    X_values, y_values = datasets[t]["test"]
    n = len(y_values)
    w = 1.0 / n  # weight for each sample in this topic/date
    sample_weights.append(np.full(n, w))
sample_weights_full = np.concatenate(sample_weights)

# replay buffer initialization
replay_buffer = ReplayBuffer(samples_per_task=DIM, seed=42)

# sequential training
for i, (task, data) in enumerate(datasets.items()):
    print(f"\n=== Phase {i+1}: Training/Fine-tuning on {TASK_TYPE}: {task} ===")

    train_dataset, y_train = data["train"]
    val_dataset, y_val = data["val"]
    test_dataset, y_test = data["test"]

    # Add replay data to current training data
    train_buffer, _ = replay_buffer.get()
    if train_buffer is not None:
        # Concatenate current data with replay memory
        train_dataset_combined = concatenate_datasets([train_dataset, train_buffer])
        train_dataset_combined = train_dataset_combined.shuffle(seed=42)
    else:
        train_dataset_combined = train_dataset

    # define trainer
    trainer = Trainer(
        model=best_model,
        args=train_args,
        train_dataset=train_dataset_combined,
        eval_dataset=val_dataset,
    )

    # fine-tune on train + val
    trainer.train()

    # Add training data to replay buffer
    replay_buffer.add(task, train_dataset, y_train)
    print(f"Task '{task}' added to replay buffer.\n")

    # evaluate on current dataset
    y_pred = trainer.predict(test_dataset)
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
    
    # evaluation on cumulative test set
    print(f"\n--- Evaluation on cumulative test set up to {TASK_TYPE} {task} ---")
    current_n = len(y_test)
    current_w = 1.0 / current_n
    current_sample_weights = np.full(current_n, current_w) # weights for each sample in current topic/date
    # add current test data and weights to cumulative lists
    X_cumulative_test.append(test_dataset)
    y_cumulative_test.append(y_test)
    sample_weights_cumulative.append(current_sample_weights) 
    # concatenate cumulative test data and weights
    X_cum_test_concat = concatenate_datasets(X_cumulative_test)
    y_cum_test_concat = np.concatenate(y_cumulative_test)
    sample_weights_cum_concat = np.concatenate(sample_weights_cumulative)
    # predict and evaluate
    preds_cumulative = trainer.predict(X_cum_test_concat)
    preds_cumulative = np.argmax(preds_cumulative.predictions, axis=1)
    results_cumulative[task] = f1_score(y_cum_test_concat, preds_cumulative, average="weighted", sample_weight=sample_weights_cum_concat)
    print(f"Evaluation on cumulative test set after {TASK_TYPE} {task}: Weighted F1 = {results_cumulative[task]:.4f}")

    # evaluation on full test set
    print("\n--- Evaluation on full test set ---")
    preds_full = trainer.predict(X_full_test)
    preds_full = np.argmax(preds_full.predictions, axis=1)
    f1_full = f1_score(y_full_test, preds_full, average="weighted", sample_weight=sample_weights_full)
    results_full[task] = f1_full
    print(f"Evaluation on full test set: Weighted F1 = {f1_full:.4f}")



# ---------------
# Results summary
# ---------------

print_results_summary(results_task, results_full, type=TASK_TYPE.capitalize())

# save results for Single Topic/Date
path = f"{RESULTS_DIR_PATH}/results_single_{TASK_TYPE}_{DIM}.json"
with open(path, "w") as f:
    json.dump(results_task, f, indent=4)

# save results for Cumulative Test Set
path = f"{RESULTS_DIR_PATH}/results_cumulative_{TASK_TYPE}_{DIM}.json"
with open(path, "w") as f:
    json.dump(results_cumulative, f, indent=4)

# save results for Full Test Set
path = f"{RESULTS_DIR_PATH}/results_full_{TASK_TYPE}_{DIM}.json"
with open(path, "w") as f:
    json.dump(results_full, f, indent=4)



# -----------------------------------------
# Plot on each Topic/Date and Full Test Set
# -----------------------------------------

path = f"{RESULTS_DIR_PATH}/f1_by_{TASK_TYPE}_{DIM}.png"
plot_f1_by_task(results_task, type=TASK_TYPE.capitalize(), path=path)

path = f"{RESULTS_DIR_PATH}/f1_cumulative_test_{TASK_TYPE}_{DIM}.png"
plot_f1_cumulative_test(results_cumulative, type=TASK_TYPE.capitalize(), path=path)

path = f"{RESULTS_DIR_PATH}/f1_full_test_{TASK_TYPE}_{DIM}.png"
plot_f1_full_test(results_full, type=TASK_TYPE.capitalize(), path=path)