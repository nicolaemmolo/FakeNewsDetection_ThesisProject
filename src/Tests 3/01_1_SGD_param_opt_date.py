from utils.utils_data_preprocessing import *
from utils.utils import *

import pandas as pd
from sklearn.metrics import f1_score
import json

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

import optuna
from functools import partial
from sklearn.utils import shuffle

# Ignore warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

RESULTS_DIR_PATH = "results/results_SGD_date"
bool_150 = False


# -----------------------
# Model building function
# -----------------------

def build_model(loss="hinge", penalty="l2", alpha=1e-4, learning_rate="optimal", eta0=0.001):
    """
    Builds a scikit-learn Pipeline with TF-IDF vectorization and SGDClassifier (incremental SVM-like model).

    Args:
        loss (str): Loss function
        penalty (str): Regularization type
        alpha (float): Regularization strength
        learning_rate (str): Learning rate schedule
        eta0 (float): Initial learning rate

    Returns:
        Pipeline: TF-IDF + SGDClassifier pipeline.
    """
    return Pipeline([
        ('tfidf', TfidfVectorizer(  # TD-IDF vectorization inside pipeline
            max_features=5000,      # limit to top 5000 features
            ngram_range=(1, 2),     # unigrams + bigrams
            stop_words="english"    # remove English stop words
        )),
        ('clf', SGDClassifier(      # SGD Classifier
            loss=loss,
            penalty=penalty,
            alpha=alpha,
            learning_rate=learning_rate,
            eta0=eta0,
            max_iter=5,
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

    loss = trial.suggest_categorical("loss", ["hinge", "log_loss", "modified_huber", "squared_hinge"])
    penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
    alpha = trial.suggest_categorical("alpha", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
    learning_rate = trial.suggest_categorical("learning_rate", ["optimal", "constant", "adaptive", "invscaling"])
    eta0 = trial.suggest_categorical("eta0", [0.0001, 0.001, 0.01, 0.1])

    model = build_model(
        loss=loss,
        penalty=penalty,
        alpha=alpha,
        learning_rate=learning_rate,
        eta0=eta0
    )

    try:
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        f1 = f1_score(y_val, preds, average="weighted") # weighted F1-score: average for label imbalance
        return f1

    except Exception as e:
        return 0.0 # return a default value in case of failure



# ----------------
# Data Preparation
# ----------------

dataset_df = data_by_date()

print("\n--- DATASET SIZES BY DATE ---")
for date, df in dataset_df.items():
    print(f"Date: {date}, Number of samples: {len(df)}")

if bool_150: # split all datasets in train/val/test (max 150 samples in val and test)
    datasets = {date: split_dataset_150(df) for date, df in dataset_df.items()}
else: # split all datasets in train/val/test
    datasets = {date: split_dataset(df) for date, df in dataset_df.items()} 

print("--- DATA SPLITS SIZES BY DATE ---")
for date, data_splits in datasets.items():
    print(f"Date: {date}")
    print(f"Train: {len(data_splits['train'][0])}, Val: {len(data_splits['val'][0])}, Test: {len(data_splits['test'][0])}")



# ---------------------------
# Hyperparameter optimization
# ---------------------------

# merge all training and validation data across topics for optimization
X_train = pd.concat([datasets[topic]['train'][0] for topic in datasets]).reset_index(drop=True)
y_train = pd.concat([datasets[topic]['train'][1] for topic in datasets]).reset_index(drop=True)
X_val = pd.concat([datasets[topic]['val'][0] for topic in datasets]).reset_index(drop=True)
y_val = pd.concat([datasets[topic]['val'][1] for topic in datasets]).reset_index(drop=True)
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
study.optimize(objective_with_data, n_trials=100) # 100 trials for demonstration; increase for better results

print("Best parameters:", study.best_params)

path = f"{RESULTS_DIR_PATH}/best_params"
if bool_150: path += "_150" # add "_150" suffix to path, if bool_150 is True
with open(f"{path}.json", "w") as f:
    json.dump(study.best_params, f, indent=4)