from utils.utils_data_preprocessing import *
from utils.utils import *

import pandas as pd
from sklearn.metrics import f1_score
import json

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

import optuna
from functools import partial
from sklearn.utils import shuffle

# Ignore warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

RESULTS_DIR_PATH = "results/results_PA_topic"
bool_150 = False



# -----------------------
# Model building function
# -----------------------

def build_model(C=1.0, loss='hinge', average=False, fit_intercept=True, shuffle=True, class_weight=None, random_state=42):
    """
    Builds a scikit-learn Pipeline with TF-IDF vectorization and PassiveAggressiveClassifier.

    Args:
        C (float): Regularization parameter for PassiveAggressiveClassifier.
        loss (str): Loss function to use.
        average (bool): Whether to use averaging.
        fit_intercept (bool): Whether to fit the intercept.
        shuffle (bool): Whether to shuffle the training data.
        class_weight (dict or 'balanced' or None): Weights associated with classes.
        random_state (int): Random seed for reproducibility.

    Returns:
        Pipeline: TF-IDF + PassiveAggressiveClassifier pipeline.
    """
    return Pipeline([
        ('tfidf', TfidfVectorizer(  # TF-IDF vectorization inside pipeline
            max_features=5000,      # limit to top 5000 features
            ngram_range=(1, 2),     # unigrams + bigrams
            stop_words='english'    # remove English stop words
        )),
        ('clf', PassiveAggressiveClassifier( # Passive Aggressive Classifier
            C=C,
            loss=loss,
            average=average,
            fit_intercept=fit_intercept,
            shuffle=shuffle,
            class_weight=class_weight,
            random_state=random_state
        ))
    ])



# -------------------------
# Optuna objective function
# -------------------------

def objectivePA(trial, X_train, y_train, X_val, y_val):
    """
    Objective function for Optuna hyperparameter optimization of PassiveAggressiveClassifier.

    Args:
        trial (optuna.trial.Trial): Optuna trial object.
        X_train (pd.Series): Training features.
        y_train (pd.Series): Training labels.
        X_val (pd.Series): Validation features.
        y_val (pd.Series): Validation labels.

    Returns:
        float: Weighted F1-score on validation set.
    """

    C = trial.suggest_categorical("C", [0.01, 0.1, 1.0, 10.0, 100.0])
    loss = trial.suggest_categorical("loss", ["hinge", "squared_hinge"])
    average = trial.suggest_categorical("average", [True, False])
    fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])
    shuffle = trial.suggest_categorical("shuffle", [True, False])
    class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])

    model = build_model(
        C=C,
        loss=loss,
        average=average,
        fit_intercept=fit_intercept,
        shuffle=shuffle,
        class_weight=class_weight
    )

    try:
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        f1 = f1_score(y_val, preds, average='weighted') # weighted F1-score: average for label imbalance
        return f1
    
    except Exception as e:
        return 0.0  # return a default value in case of failure



# ----------------
# Data Preparation
# ----------------

dataset_df = data_by_topic()

print("--- DATASET SIZES BY TOPIC ---")
for topic, df in dataset_df.items():
    print(f"Topic: {topic}, Number of samples: {len(df)}")

if bool_150: # split all datasets in train/val/test (max 150 samples in val and test)
    datasets = {topic: split_dataset_150(df) for topic, df in dataset_df.items()}
else: # split all datasets in train/val/test
    datasets = {topic: split_dataset(df) for topic, df in dataset_df.items()}

print("--- DATA SPLITS SIZES BY TOPIC ---")
for topic, data_splits in datasets.items():
    print(f"Topic: {topic}")
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
objective_with_data = partial(objectivePA,
                              X_train=X_train,
                              X_val=X_val,
                              y_train=y_train,
                              y_val=y_val)

study = optuna.create_study(direction="maximize") # maximize F1-score
study.optimize(objective_with_data, n_trials=100) # 100 trials for demonstration; increase for better results

print("Best parameters:", study.best_params)

path = f"{RESULTS_DIR_PATH}/best_params"
if bool_150: path += "_150"
with open(f"{path}.json", "w") as f:
    json.dump(study.best_params, f, indent=4)