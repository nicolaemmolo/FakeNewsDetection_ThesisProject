from utils.utils_data_preprocessing import *
from utils.utils import *

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import json

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

from sklearn.utils import shuffle

# Ignore warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

TASK_TYPE = "topic" # specify the task type ("topic" or "date")
RESULTS_DIR_PATH = f"results/results_SGD_{TASK_TYPE}"
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



# -----------------------
# Model building function
# -----------------------

def build_model(loss="hinge", penalty="l2", alpha=1e-4, learning_rate="optimal", eta0=0.001):
    """
    Builds a scikit-learn Pipeline with Hashing vectorization and SGDClassifier (incremental SVM-like model).

    Args:
        loss (str): Loss function
        penalty (str): Regularization type
        alpha (float): Regularization strength
        learning_rate (str): Learning rate schedule
        eta0 (float): Initial learning rate

    Returns:
        Pipeline: Vectorizer + SGDClassifier pipeline.
    """
    return Pipeline([
        ('vect', HashingVectorizer( # Hashing vectorization
            n_features=2**20,        # 1 milion of features: reduces collisions drastically for news
            ngram_range=(1, 2),      # Unigrams and Bigrams (essential to understand context in news)
            stop_words="english"     # Removes useless words
        )),
        ('clf', SGDClassifier(       # SGD Classifier
            loss=loss,
            penalty=penalty,
            alpha=alpha,
            learning_rate=learning_rate,
            eta0=eta0
        ))
    ])



# ----------------
# Data Preparation
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

if bool_150: # split all datasets in train/val/test (max 150 samples in val and test)
    datasets = {task: split_dataset_150(df) for task, df in dataset_df.items()}
else: # split all datasets in train/val/test
    datasets = {task: split_dataset(df) for task, df in dataset_df.items()} 

print(f"--- DATA SPLITS SIZES BY {TASK_TYPE.upper()} ---")
for task, data_splits in datasets.items():
    print(f"{TASK_TYPE.capitalize()}: {task}")
    print(f"Train: {len(data_splits['train'][0])}, Val: {len(data_splits['val'][0])}, Test: {len(data_splits['test'][0])}")



# ------------------------------------
# Fine-tuning on Dataset by Topic/Date
# ------------------------------------

path = f"{RESULTS_DIR_PATH}/best_params"
if bool_150: path += "_150" # add "_150" suffix to path, if bool_150 is True
with open(f"{path}.json", "r") as f:
    best_params = json.load(f)

best_model = build_model(
    loss=best_params["loss"],
    penalty=best_params["penalty"],
    alpha=best_params["alpha"],
    learning_rate=best_params["learning_rate"],
    eta0=best_params["eta0"]
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
    X_full_test = pd.concat(X_trimmed_list).reset_index(drop=True)
    y_full_test = pd.concat(y_trimmed_list).reset_index(drop=True)
elif bool_balanced_test: # balanced full test set (via sample weights)
    X_full_test = pd.concat([datasets[t]["test"][0] for t in datasets]).reset_index(drop=True)
    y_full_test = pd.concat([datasets[t]["test"][1] for t in datasets]).reset_index(drop=True)
    sample_weights = []
    for t in datasets:
        _, y_test_t = datasets[t]["test"]
        n = len(y_test_t)
        w = 1.0 / n  # weight for each sample in this topic/date
        sample_weights.extend([w] * n)
    sample_weights = np.array(sample_weights)
else: # unbalanced full test set
    X_full_test = pd.concat([datasets[t]["test"][0] for t in datasets]).reset_index(drop=True)
    y_full_test = pd.concat([datasets[t]["test"][1] for t in datasets]).reset_index(drop=True)

# sequential training
for i, (task, data) in enumerate(datasets.items()):
    print(f"\n=== Phase {i+1}: Training/Fine-tuning on {TASK_TYPE}: {task} ===")
    
    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    X_test, y_test = data["test"]
        
    # vectorization of Train and Validation sets
    X_train = best_model.named_steps['vect'].transform(X_train)
    X_val = best_model.named_steps['vect'].transform(X_val)
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)

    # early stopping settings
    n_epochs = 50        # max number of epochs
    patience = 3         # number of epochs to wait without improvement before stopping
    best_score = -np.inf # initialize the best score to the lowest possible value
    wait = 0             # counter for "patience"

    for epoch in range(n_epochs):
        # Training Epoch
        X_train, y_train = shuffle(X_train, y_train, random_state=None)
        best_model.named_steps['clf'].partial_fit(
            X_train,
            y_train,
            classes=[0,1]
        )
        # Early Stopping check
        current_score = best_model.named_steps['clf'].score(X_val, y_val)
        if current_score > best_score: # improvement
            best_score = current_score
            wait = 0
        else: # no improvement
            wait += 1
            if wait >= patience:
                break

    y_pred = best_model.predict(X_test)
    print(f"Classification Report after {TASK_TYPE} {task}:")
    print(classification_report(y_test, y_pred))
    print(f"Confusion Matrix after {TASK_TYPE} {task}:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\nWeighted F1-score after {TASK_TYPE} {task}:", f1_score(y_test, y_pred, average="weighted"))

    # evaluation on all topics/dates
    print(f"\n--- Evaluation on all {TASK_TYPE}s ---")
    results_task[task] = {}
    for test_task, test_data in datasets.items(): # for each topic/date
        X_te, y_te = test_data["test"]
        preds = best_model.predict(X_te)
        f1 = f1_score(y_te, preds, average="weighted")
        results_task[task][test_task] = f1
        print(f"Evaluation on {TASK_TYPE} {test_task}: Weighted F1 = {f1:.4f}")

    # evaluation on full test set
    print("\n--- Evaluation on full test set ---")
    preds_full = best_model.predict(X_full_test)
    if bool_balanced_test:
        f1_full = f1_score(y_full_test, preds_full, average="weighted", sample_weight=sample_weights)
    else:
        f1_full = f1_score(y_full_test, preds_full, average="weighted")
    results_full[task] = f1_full
    print(f"Evaluation on full test set after {TASK_TYPE} {task}: Weighted F1 = {f1_full:.4f}")



# ---------------
# Results summary
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