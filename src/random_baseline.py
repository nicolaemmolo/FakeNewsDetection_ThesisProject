import os

from utils.utils_data_preprocessing import *
from utils.utils import *

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import json

TASK_TYPE = "topic" # specify the task type ("topic" or "date")
MAIN_DIR_PATH = "results_dist_rep"
RESULTS_DIR_PATH = f"{MAIN_DIR_PATH}/results_random_{TASK_TYPE}"
os.makedirs(RESULTS_DIR_PATH, exist_ok=True)


# ---------------
# Random baseline
# ---------------

def random_baseline_f1(y_true, sample_weight=None, n_runs=100):
    rng = np.random.RandomState(42)
    y_true = np.array(y_true)

    # empirical class prior
    p_pos = np.mean(y_true)

    scores = []
    for _ in range(n_runs):
        y_pred = rng.choice(
            [0,1],
            size=len(y_true),
            p=[1-p_pos, p_pos]
        )
        score = f1_score(y_true, y_pred, average="weighted", sample_weight=sample_weight)
        scores.append(score)

    return float(np.mean(scores)), float(np.std(scores))



# ----------------
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



# ---------------------------------------------
# Random baseline evaluation on each topic/date
# ---------------------------------------------

results = {}
for i, (task, data) in enumerate(datasets.items()):
    _, y_test = data["test"]

    mean_f1, std_f1 = random_baseline_f1(y_test)

    print(f"\n--- RANDOM BASELINE PERFORMANCE FOR {TASK_TYPE.upper()}: {task} ---")
    print(f"Task type: {TASK_TYPE}")
    print(f"Weighted F1-score: {mean_f1:.4f} ± {std_f1:.4f}")

    results[task] = {
        "mean": mean_f1,
        "std": std_f1
    }

path = f"{RESULTS_DIR_PATH}/random_baseline_by_{TASK_TYPE}.json"
with open(path, "w") as f:
    json.dump(results, f, indent=4)



# -------------------------------------------
# Random baseline evaluation on full test set
# -------------------------------------------

# test set for balanced full evaluation (via sample weights)
X_full_test = pd.concat([datasets[t]["test"][0] for t in datasets]).reset_index(drop=True)
y_full_test = pd.concat([datasets[t]["test"][1] for t in datasets]).reset_index(drop=True)
# sample weights (balanced across tasks)
sample_weights = []
for t in datasets:
    _, y_t = datasets[t]["test"]
    n = len(y_t)
    w = 1.0 / n
    sample_weights.append(np.full(n, w))
sample_weights_full = np.concatenate(sample_weights)

# evaluate random baseline on full test set
mean_f1, std_f1 = random_baseline_f1(y_full_test, sample_weight=sample_weights_full)

print("\n--- RANDOM BASELINE PERFORMANCE ---")
print(f"Task type: {TASK_TYPE}")
print(f"Weighted F1-score: {mean_f1:.4f} ± {std_f1:.4f}")

results = {
    "mean": mean_f1,
    "std": std_f1
}

path = f"{RESULTS_DIR_PATH}/random_baseline_full_test.json"
with open(path, "w") as f:
    json.dump(results, f, indent=4)