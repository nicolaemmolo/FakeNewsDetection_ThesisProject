from utils.utils_data_preprocessing import *
from utils.utils import *

import pandas as pd
from sklearn.metrics import f1_score
import json

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

import optuna
from functools import partial
from sklearn.utils import shuffle

# Ignore warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

TASK_TYPE = "date" # specify the task type ("topic" or "date")
RESULTS_DIR_PATH = f"results/results_Linear_{TASK_TYPE}"
bool_150 = False


best_params = {}
path = f"{RESULTS_DIR_PATH}/best_params"
if bool_150: path += "_150" # add "_150" suffix to path, if bool_150 is True
with open(f"{path}.json", "r") as f:
    best_params = json.load(f)


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
            eta0=eta0,
            max_iter=1000,
            early_stopping=True,
            n_iter_no_change=5,
            random_state=42
        ))
    ])



# -------------------------
# Optuna objective function
# -------------------------

def objectiveSGD(trial, X_train, y_train, X_val, y_val):
    """
    Objective function for Optuna hyperparameter optimization of SGDClassifier model.

    Args:
        trial: An Optuna trial object for suggesting hyperparameters
        X_train: Training feature data
        y_train: Training labels
        X_val: Validation feature data
        y_val: Validation labels
    
    Returns:
        float: The weighted F1-score on the validation set
    """


    model = build_model(
        loss = best_params["loss"],
        penalty=best_params["penalty"],
        alpha=best_params["alpha"],
        learning_rate=best_params["learning_rate"],
        eta0=best_params["eta0"]
    )

    try:
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        f1 = f1_score(y_val, preds, average="weighted") # weighted F1-score: average for label imbalance
        return f1

    except Exception as e:
        raise optuna.exceptions.TrialPruned()  # prune the trial in case of failure 



# ----------------
# Data Preparation
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

if bool_150: # split all datasets in train/val/test (max 150 samples in val and test)
    datasets = {task: split_dataset_150(df) for task, df in dataset_df.items()}
else: # split all datasets in train/val/test
    datasets = {task: split_dataset(df) for task, df in dataset_df.items()} 

print(f"--- DATA SPLITS SIZES BY {TASK_TYPE.upper()} ---")
for task, data_splits in datasets.items():
    print(f"{TASK_TYPE.capitalize()}: {task}")
    print(f"Train: {len(data_splits['train'][0])}, Val: {len(data_splits['val'][0])}, Test: {len(data_splits['test'][0])}")



# ---------------------------
# Hyperparameter optimization
# ---------------------------

# merge all training and validation data across topics/dates for optimization
X_train = pd.concat([datasets[task]['train'][0] for task in datasets]).reset_index(drop=True)
y_train = pd.concat([datasets[task]['train'][1] for task in datasets]).reset_index(drop=True)
X_val = pd.concat([datasets[task]['val'][0] for task in datasets]).reset_index(drop=True)
y_val = pd.concat([datasets[task]['val'][1] for task in datasets]).reset_index(drop=True)
# shuffle the combined training and validation data
X_train, y_train = shuffle(X_train, y_train, random_state=42)
X_val, y_val = shuffle(X_val, y_val, random_state=42)

# functools.partial to pass data to objective function
objective_with_data = partial(objectiveSGD,
                              X_train=X_train,
                              X_val=X_val,
                              y_train=y_train,
                              y_val=y_val)

study = optuna.create_study(direction="maximize") # maximize F1-score
study.optimize(objective_with_data, n_trials=1) # 100 trials for demonstration; increase for better results

print("Best parameters:", study.best_params)