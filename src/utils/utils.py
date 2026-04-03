import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



# ---------------------------
# Dataset splitting functions
# ---------------------------

def split_dataset(df, test_size=0.2, val_size=0.2, random_state=42):
    """
    Split the dataset into training, validation, and test sets.

    Args:
        df (pd.DataFrame): DataFrame containing 'texts' and 'labels'
        test_size (float): Proportion of the dataset to include in the test split.
        val_size (float): Proportion of the dataset to include in the validation split.
        random_state (int): Random seed for reproducibility.

    Returns:
        dict: A dictionary with keys 'train', 'val', 'test' and values as tuples of (X, y)
    """    

    X = df['texts'].astype(str)
    y = df['labels']

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), stratify=y, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=random_state
    )

    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test)
    }

def split_dataset_tensorflow(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """
    Split the dataset into training, validation, and test sets.

    Args:
        X (np.array): Features
        y (np.array): Labels
        test_size (float): Proportion of the dataset to include in the test split.
        val_size (float): Proportion of the dataset to include in the validation split.
        random_state (int): Random seed for reproducibility.

    Returns:
        dict: A dictionary with keys 'train', 'val', 'test' and values as tuples of (X, y)
    """    

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), stratify=y, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=random_state
    )

    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test)
    }

def split_dataset_150(df, test_size=0.2, val_size=0.2, random_state=42):
    """
    Split the dataset into training, validation, and test sets with max length of 150 for val and test.

    Args:
        df (pd.DataFrame): DataFrame containing 'texts' and 'labels'
        test_size (float): Proportion of the dataset to include in the test split.
        val_size (float): Proportion of the dataset to include in the validation split.
        random_state (int): Random seed for reproducibility.

    Returns:
        dict: A dictionary with keys 'train', 'val', 'test' and values as tuples of (X, y)
    """

    data = split_dataset(df, test_size=test_size, val_size=val_size, random_state=random_state)

    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    X_test, y_test = data["test"]

    # Process VAL: trim_and_extract is the helper function to trim and extract extra data
    X_val, y_val, X_val_extra, y_val_extra = trim_and_extract(X_val, y_val)
    if X_val_extra is not None:
        X_train = pd.concat([X_train, X_val_extra], ignore_index=True)
        y_train = pd.concat([y_train, y_val_extra], ignore_index=True)

    # Process TEST: trim_and_extract is the helper function to trim and extract extra data
    X_test, y_test, X_test_extra, y_test_extra = trim_and_extract(X_test, y_test)
    if X_test_extra is not None:
        X_train = pd.concat([X_train, X_test_extra], ignore_index=True)
        y_train = pd.concat([y_train, y_test_extra], ignore_index=True)

    return {
        "train": (X_train.reset_index(drop=True), y_train.reset_index(drop=True)),
        "val": (X_val.reset_index(drop=True), y_val.reset_index(drop=True)),
        "test": (X_test.reset_index(drop=True), y_test.reset_index(drop=True)),
    }

def split_dataset_tensorflow_150(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """
    Split the dataset into training, validation, and test sets with max length of 150 for val and test.

    Args:
        X (np.array): Features
        y (np.array): Labels
        test_size (float): Proportion of the dataset to include in the test split.
        val_size (float): Proportion of the dataset to include in the validation split.
        random_state (int): Random seed for reproducibility.

    Returns:
        dict: A dictionary with keys 'train', 'val', 'test' and values as tuples of (X, y)
    """
    
    data = split_dataset_tensorflow(X, y, test_size=test_size, val_size=val_size, random_state=random_state)

    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    X_test, y_test = data["test"]

    # Process VAL: trim_and_extract is the helper function to trim and extract extra data
    X_val, y_val, X_val_extra, y_val_extra = trim_and_extract(X_val, y_val)
    if X_val_extra is not None:
        X_train = np.concatenate([X_train, X_val_extra], axis=0)
        y_train = np.concatenate([y_train, y_val_extra], axis=0)

    # Process TEST: trim_and_extract is the helper function to trim and extract extra data
    X_test, y_test, X_test_extra, y_test_extra = trim_and_extract(X_test, y_test)
    if X_test_extra is not None:
        X_train = np.concatenate([X_train, X_test_extra], axis=0)
        y_train = np.concatenate([y_train, y_test_extra], axis=0)

    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }

def trim_and_extract(X, y, max_len=150):
    """
    Trims the dataset to max_len and extracts the remainder.
    
    Args:
        X (pd.Series): Series containing text data.
        y (pd.Series): Series containing label data.
        max_len (int): Maximum length for the dataset.
        
    Returns:
        tuple: A tuple containing trimmed X, trimmed y, extra X, and extra y.
    """
    if len(X) <= max_len:
        return X, y, None, None

    # Trim X
    if isinstance(X, pd.Series):
        X_trim = X.iloc[:max_len]
        X_extra = X.iloc[max_len:]
    elif isinstance(X, np.ndarray):
        X_trim = X[:max_len]
        X_extra = X[max_len:] 
    else:
        # HuggingFace Dataset
        X_trim = X.select(range(max_len))
        X_extra = X.select(range(max_len, len(X)))

    # Trim y
    if isinstance(y, pd.Series):
        y_trim = y.iloc[:max_len]
        y_extra = y.iloc[max_len:]
    elif isinstance(y, np.ndarray):
        y_trim = y[:max_len]
        y_extra = y[max_len:]
    else:
        raise TypeError(f"Unsupported y type: {type(y)}")

    return X_trim, y_trim, X_extra, y_extra



# -----------------
# Results and Plots
# -----------------

def print_results_summary(results, results_full, type="topic"):
    """
    Prints a summary of the results.

    Args:
        results (dict): Nested dictionary in form {finetuning_task: {test_task: f1_score}}.
        results_full (dict): Dictionary in form {finetuning_task: f1_score_on_full_test}.
        type (str): Type of task.
    """
    print("\n=== Results Summary ===")
    for task, res in results.items():
        print(f"\nResults after training on {type} {task}:")
        for test_task, f1 in res.items():
            print(f"  Test on {type} {test_task}: Weighted F1 = {f1:.4f}")

    print("\n=== Full Test Set Results Summary ===")
    for task, f1 in results_full.items():
        print(f"After training on {type} {task}: Full Test Set Weighted F1 = {f1:.4f}")


def create_path_suffix(bool_150, bool_150test, bool_balanced_test, bool_inverted=False):
    """
    Creates a suffix for file paths based on boolean flags.

    Args:
        bool_150 (bool): Flag for 150-length datasets.
        bool_150test (bool): Flag for 150-length test datasets.
        bool_balanced_test (bool): Flag for balanced test datasets.
        bool_inverted (bool): Flag for inverted datasets.
    
    Returns:
        str: The constructed path suffix.
    """
    path = ""
    if bool_150:
        path += "_150"
    if bool_150test:
        path += "_150test"
    if bool_balanced_test:
        path += "_balancedtest"
    if bool_inverted:
        path += "_inverted"
    return path


def plot_f1_by_task(results_task, type, path):
    """
    Plots the F1-score for each task test set as a function of the fine-tuning task.

    Args:
        results_task (dict): Nested dictionary in form {finetuning_task: {test_task: f1_score}}.
        type (str): Type of task.
        path (str): Path to save the plot image.
    """

    # extract tasks
    tasks = list(results_task.keys())
    test_tasks = list(next(iter(results_task.values())).keys())

    # make F1-score matrix
    f1_matrix = np.array([[results_task[train_task][test_task] for test_task in test_tasks] for train_task in tasks])
    plt.figure(figsize=(10, 6))

    # plot each test task line
    for i, test_task in enumerate(test_tasks):
        plt.plot(tasks, f1_matrix[:, i], marker='o', label=f"Test on {test_task}")

    plt.title(f"F1-score on each {type} Test Set")
    plt.xlabel(f"Fine-tuning {type}")
    plt.ylabel("Weighted F1-score")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    plt.savefig(path)
    plt.close()


def plot_f1_cumulative_test(results_cumulative, type, path):
    """
    Plots the F1-score on the cumulative test set as a function of the fine-tuning task.

    Args:
        results_cumulative (dict): Dictionary in form {finetuning_task: f1_score_on_cumulative_test}.
        type (str): Type of task.
        path (str): Path to save the plot image.
    """

    # extract tasks
    tasks = list(results_cumulative.keys())

    plt.figure(figsize=(6, 4))

    # plot F1-score on cumulative test set
    plt.plot(tasks, [results_cumulative[task] for task in tasks], marker='o', color='g')

    plt.title(f"F1-score on Cumulative Test Set (Datasets by {type})")
    plt.xlabel(f"Fine-tuning {type}")
    plt.ylabel("Weighted F1-score")
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    plt.savefig(path)
    plt.close()


def plot_f1_full_test(results_full, type, path):
    """
    Plots the F1-score on the full test set as a function of the fine-tuning task.

    Args:
        results_full (dict): Dictionary in form {finetuning_task: f1_score_on_full_test}.
        type (str): Type of task.
        path (str): Path to save the plot image.
    """

    # extract tasks
    tasks = list(results_full.keys())

    plt.figure(figsize=(6, 4))

    # plot F1-score on full test set
    plt.plot(tasks, [results_full[task] for task in tasks], marker='o', color='b')

    plt.title(f"F1-score on Full Test Set (Datasets by {type})")
    plt.xlabel(f"Fine-tuning {type}")
    plt.ylabel("Weighted F1-score")
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    plt.savefig(path)
    plt.close()


def plot_fisher_histogram(filename, type, path):
    # 1. Caricamento dati
    # allow_pickle=True a volte serve, ma con .npz semplice spesso non è necessario
    data = np.load(filename, allow_pickle=True)
    
    # Estraiamo i vettori in ordine (task_0, task_1, ecc.)
    history = []
    # I file .npz si comportano come dizionari. Le chiavi sono i nomi che abbiamo dato prima.
    for key in sorted(data.files):
        history.append(data[key])
    
    # 2. Creazione Istogramma (Codice che ti ho dato prima)
    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(history)))

    for i, importance_vector in enumerate(history):
        # Ottimizzazione: Usiamo float16 per risparmiare RAM durante il plot se i dati sono enormi
        vals = importance_vector.astype(np.float16)
        
        plt.hist(
            vals, 
            bins=100, 
            alpha=0.5, 
            label=f'End Task {i+1}', 
            color=colors[i],
            log=True # Scala Logaritmica fondamentale!
        )

    plt.title(f"Fisher Importance Evolution ({type})")
    plt.xlabel("Fisher Importance Values")
    plt.ylabel("Parameter Count (Log)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(path)
    plt.close()
