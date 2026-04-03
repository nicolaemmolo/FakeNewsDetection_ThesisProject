import os

from utils.utils_data_preprocessing import *
from utils.utils import *

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import json

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.utils import shuffle

# Ignore warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

TASK_TYPE = "date" # specify the task type ("topic" or "date")
MODEL = "Linear" # specify the model type ("Linear", "NB", or "PA")
RESULTS_DIR_PATH = f"results_finetuning/results_{MODEL}_{TASK_TYPE}"
os.makedirs(RESULTS_DIR_PATH, exist_ok=True)



# ------------------------
# Model building functions
# ------------------------

def build_model(model, **kwargs):
    """
    Builds a scikit-learn Pipeline based on the specified model type.

    Args:
        model (str): Type of model to build ("Linear", "NB", or "PA").
        **kwargs: Additional parameters for the specific model building function.
    
    Returns:
        Pipeline: Vectorizer + Classifier pipeline.
    """
    if model == "Linear":
        return build_model_linear(**kwargs)
    elif model == "NB":
        return build_model_nb(**kwargs)
    elif model == "PA":
        return build_model_pa(**kwargs)
    else:
        raise ValueError("Invalid model type. Choose either 'Linear', 'NB', or 'PA'.")

def build_model_linear(loss="hinge", penalty="l2", alpha=1e-4, learning_rate="optimal", eta0=0.001):
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

def build_model_nb(alpha=1.0, fit_prior=False):
    """
    Builds a scikit-learn Pipeline with Hashing vectorization and Multinomial Naive Bayes classifier.

    Args:
        alpha (float): Smoothing parameter for Multinomial Naive Bayes.
        fit_prior (bool): Whether to learn class prior probabilities.

    Returns:
        Pipeline: Vectorizer + Multinomial Naive Bayes pipeline.
    """
    return Pipeline([
        ('vect', HashingVectorizer( # Hashing vectorization
            n_features=2**20,       # 1 milion of features: reduces collisions drastically for news
            ngram_range=(1, 2),     # Unigrams and Bigrams (essential to understand context in news)
            stop_words="english",   # Removes useless words
            alternate_sign=False    # To ensure non-negative feature values for Naive Bayes
        )),
        ('clf', MultinomialNB(      # Multinomial Naive Bayes classifier
            alpha=alpha,
            fit_prior=fit_prior
        ))
    ])

def build_model_pa(C=1.0, loss='hinge', average=False, fit_intercept=True, class_weight=None):
    """
    Builds a scikit-learn Pipeline with Hashing vectorization and PassiveAggressiveClassifier.

    Args:
        C (float): Regularization parameter for PassiveAggressiveClassifier.
        loss (str): Loss function to use.
        average (bool): Whether to use averaging.
        fit_intercept (bool): Whether to fit the intercept.
        class_weight (dict or 'balanced' or None): Weights associated with classes.

    Returns:
        Pipeline: Vectorizer + PassiveAggressiveClassifier pipeline.
    """
    return Pipeline([
        ('vect', HashingVectorizer( # Hashing vectorization inside pipeline
            n_features=2**20,       # 1 milion of features: reduces collisions drastically for news
            ngram_range=(1, 2),     # unigrams + bigrams
            stop_words='english'    # remove English stop words
        )),
        ('clf', PassiveAggressiveClassifier( # Passive Aggressive Classifier
            C=C,
            loss=loss,
            average=average,
            fit_intercept=fit_intercept,
            class_weight=class_weight
        ))
    ])



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

# split all datasets in train/val/test
datasets = {task: split_dataset(df) for task, df in dataset_df.items()} 

print(f"--- DATA SPLITS SIZES BY {TASK_TYPE.upper()} ---")
for task, data_splits in datasets.items():
    print(f"{TASK_TYPE.capitalize()}: {task}")
    print(f"Train: {len(data_splits['train'][0])}, Val: {len(data_splits['val'][0])}, Test: {len(data_splits['test'][0])}")



# ------------------------------------
# Fine-tuning on Dataset by Topic/Date
# ------------------------------------

path = f"{RESULTS_DIR_PATH}/best_params"
with open(f"{path}.json", "r") as f:
    best_params = json.load(f)

best_model = build_model(
    model=MODEL,
    **best_params
)

results_task = {}
results_cumulative = {}
results_full = {}

# test set for balanced cumulative/full evaluation (via sample weights)
X_cumulative_test = []
y_cumulative_test = []
X_full_test = pd.concat([datasets[t]["test"][0] for t in datasets]).reset_index(drop=True)
y_full_test = pd.concat([datasets[t]["test"][1] for t in datasets]).reset_index(drop=True)
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

# sequential training
for i, (task, data) in enumerate(datasets.items()):
    print(f"\n=== Phase {i+1}: Training/Fine-tuning on {TASK_TYPE}: {task} ===")
    
    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    X_test, y_test = data["test"]

    # vectorization
    X_train_vect = best_model.named_steps['vect'].transform(X_train)
    X_val_vect = best_model.named_steps['vect'].transform(X_val)

    if MODEL == "NB":
        # incremental fit (on train+val)
        X_task = pd.concat([X_train, X_val]).reset_index(drop=True)
        y_task = pd.concat([y_train, y_val]).reset_index(drop=True)
        X_task, y_task = shuffle(X_task, y_task, random_state=42)
        X_task = best_model.named_steps['vect'].transform(X_task)
        y_task = y_task.reset_index(drop=True)
        best_model.named_steps['clf'].partial_fit(
            X_task,
            y_task,
            classes=[0,1]
        )
    else:
        # early stopping settings
        n_epochs = 50        # max number of epochs
        patience = 3         # number of epochs to wait without improvement before stopping
        best_score = -np.inf # initialize the best score to the lowest possible value
        wait = 0             # counter for "patience"

        for epoch in range(n_epochs):
            # Training Epoch
            X_task, y_task = shuffle(X_train_vect, y_train, random_state=None)
            best_model.named_steps['clf'].partial_fit(
                X_task,
                y_task,
                classes=[0,1]
            )
            # Early Stopping check
            current_score = best_model.named_steps['clf'].score(X_val_vect, y_val)
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
        results_task[task][test_task] = f1_score(y_te, preds, average="weighted")
        print(f"Evaluation on {TASK_TYPE} {test_task}: Weighted F1 = {results_task[task][test_task]:.4f}")
    
    # evaluation on cumulative test set
    print(f"\n--- Evaluation on cumulative test set up to {TASK_TYPE} {task} ---")
    current_n = len(y_test)
    current_w = 1.0 / current_n
    current_sample_weights = np.full(current_n, current_w) # weights for each sample in current topic/date
    # add current test data and weights to cumulative lists
    X_cumulative_test.append(X_test)
    y_cumulative_test.append(y_test)
    sample_weights_cumulative.append(current_sample_weights) 
    # concatenate cumulative test data and weights
    X_cum_test_concat = pd.concat(X_cumulative_test).reset_index(drop=True)
    y_cum_test_concat = pd.concat(y_cumulative_test).reset_index(drop=True)
    sample_weights_cum_concat = np.concatenate(sample_weights_cumulative)
    # predict and evaluate
    preds_cumulative = best_model.predict(X_cum_test_concat)
    results_cumulative[task] = f1_score(y_cum_test_concat, preds_cumulative, average="weighted", sample_weight=sample_weights_cum_concat)
    print(f"Evaluation on cumulative test set after {TASK_TYPE} {task}: Weighted F1 = {results_cumulative[task]:.4f}")

    # evaluation on full test set
    print("\n--- Evaluation on full test set ---")
    preds_full = best_model.predict(X_full_test)
    results_full[task] = f1_score(y_full_test, preds_full, average="weighted", sample_weight=sample_weights_full)
    print(f"Evaluation on full test set after {TASK_TYPE} {task}: Weighted F1 = {results_full[task]:.4f}")



# ---------------
# Results summary
# ---------------

print_results_summary(results_task, results_full, type=TASK_TYPE.capitalize())

# save results for Single Topic/Date
path = f"{RESULTS_DIR_PATH}/results_single_{TASK_TYPE}.json"
with open(path, "w") as f:
    json.dump(results_task, f, indent=4)

# save results for Cumulative Test Set
path = f"{RESULTS_DIR_PATH}/results_cumulative_{TASK_TYPE}.json"
with open(path, "w") as f:
    json.dump(results_cumulative, f, indent=4)

# save results for Full Test Set
path = f"{RESULTS_DIR_PATH}/results_full_{TASK_TYPE}.json"
with open(path, "w") as f:
    json.dump(results_full, f, indent=4)



# -----------------------------------------
# Plot on each Topic/Date and Full Test Set
# -----------------------------------------

path = f"{RESULTS_DIR_PATH}/f1_by_{TASK_TYPE}.png"
plot_f1_by_task(results_task, type=TASK_TYPE.capitalize(), path=path)

path = f"{RESULTS_DIR_PATH}/f1_cumulative_test_{TASK_TYPE}.png"
plot_f1_cumulative_test(results_cumulative, type=TASK_TYPE.capitalize(), path=path)

path = f"{RESULTS_DIR_PATH}/f1_full_test_{TASK_TYPE}.png"
plot_f1_full_test(results_full, type=TASK_TYPE.capitalize(), path=path)