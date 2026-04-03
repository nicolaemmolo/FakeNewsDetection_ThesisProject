from utils.utils_data_preprocessing import *
from utils.utils import *

import pandas as pd
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import json

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Ignore warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

RESULTS_DIR_PATH = "results/results_NB_date"
bool_150 = False
bool_150test = False
bool_balanced_test = False
PATH_SUFFIX = create_path_suffix(
    bool_150=bool_150,
    bool_150test=bool_150test,
    bool_balanced_test=bool_balanced_test,
)



# -----------------------
# Model building function
# -----------------------

def build_model(alpha=1.0, fit_prior=False):
    """
    Builds a scikit-learn Pipeline with TF-IDF vectorization and Multinomial Naive Bayes classifier.

    Args:
        alpha (float): Smoothing parameter for Multinomial Naive Bayes.
        fit_prior (bool): Whether to learn class prior probabilities.

    Returns:
        Pipeline: TF-IDF + Multinomial Naive Bayes pipeline.
    """
    return Pipeline([
        ('tfidf', TfidfVectorizer(  # TD-IDF vectorization inside pipeline
            max_features=5000,      # limit to top 5000 features
            ngram_range=(1, 2),     # unigrams + bigrams
            stop_words="english"    # remove English stop words
        )),
        ('clf', MultinomialNB(      # Multinomial Naive Bayes classifier
            alpha=alpha,
            fit_prior=fit_prior
        ))
    ])



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



# ------------------------------
# Fine-tuning on Dataset by Date
# ------------------------------

path = f"{RESULTS_DIR_PATH}/best_params"
if bool_150: path += "_150" # add "_150" suffix to path, if bool_150 is True
with open(f"{path}.json", "r") as f:
    best_params = json.load(f)
    
best_model = build_model(
    alpha=best_params['alpha'],
    fit_prior=best_params['fit_prior']
)

results_date = {}
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
        w = 1.0 / n  # weight for each sample in this topic
        sample_weights.extend([w] * n)
    sample_weights = np.array(sample_weights)
else: # unbalanced full test set
    X_full_test = pd.concat([datasets[t]["test"][0] for t in datasets]).reset_index(drop=True)
    y_full_test = pd.concat([datasets[t]["test"][1] for t in datasets]).reset_index(drop=True)

# sequential training
for i, (date, data) in enumerate(datasets.items()):
    print(f"\n=== Phase {i+1}: Training/Fine-tuning on date: {date} ===")
    
    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    X_test, y_test = data["test"]

    # imcremental fit (on train+val)
    best_model.named_steps['clf'].partial_fit(
        best_model.named_steps['tfidf'].fit_transform(pd.concat([X_train, X_val])),
        pd.concat([y_train, y_val]),
        classes=[0, 1]  # possible classes: assuming binary classification
    )

    y_pred = best_model.predict(X_test)
    print(f"Classification Report after date {date}:")
    print(classification_report(y_test, y_pred))
    print(f"Confusion Matrix after date {date}:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\nWeighted F1-score after date {date}:", f1_score(y_test, y_pred, average="weighted"))


    # evaluation on all dates
    print("\n--- Evaluation on all dates ---")
    results_date[date] = {}
    for test_date, test_data in datasets.items(): # for each date
        X_te, y_te = test_data["test"]
        preds = best_model.predict(X_te)
        f1 = f1_score(y_te, preds, average="weighted")
        results_date[date][test_date] = f1
        print(f"Evaluation on date {test_date}: Weighted F1 = {f1:.4f}")
    
    # evaluation on full test set
    print("\n--- Evaluation on full test set ---")
    preds_full = best_model.predict(X_full_test)
    if bool_balanced_test:
        f1_full = f1_score(y_full_test, preds_full, average="weighted", sample_weight=sample_weights)
    else:
        f1_full = f1_score(y_full_test, preds_full, average="weighted")
    results_full[date] = f1_full
    print(f"Evaluation on full test set after date {date}: Weighted F1 = {f1_full:.4f}")



# ---------------
# Results summary
# ---------------

print_results_summary(results_date, results_full, type="Date")

# save results for Single Date
path = f"{RESULTS_DIR_PATH}/results_single_date{PATH_SUFFIX}.json"
with open(path, "w") as f:
    json.dump(results_date, f, indent=4)

# save results for Full Test Set
path = f"{RESULTS_DIR_PATH}/results_full_date{PATH_SUFFIX}.json"
with open(path, "w") as f:
    json.dump(results_full, f, indent=4)



# -----------------------------------
# Plot on each Date and Full Test Set
# -----------------------------------

path = f"{RESULTS_DIR_PATH}/f1_by_date{PATH_SUFFIX}.png"
plot_f1_by_task(results_date, type="Date", path=path)

path = f"{RESULTS_DIR_PATH}/f1_full_test_date{PATH_SUFFIX}.png"
plot_f1_full_test(results_full, type="Date", path=path)