import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

TOPIC_ORDER = ['politics', 'general', 'covid', 'syria', 'islam', 'notredame', 'gossip'] # Topic order for plots
DATE_ORDER = ['2011-2015', '2016', '2017', '2019', '2020'] # Date order for plots
MODEL_ORDER = ['BERT', 'DeBERTa', 'RoBERTa', 'CNN', 'BiLSTM', 'Linear', 'NB', 'PA'] # Model order for plots

MODEL_COLORS = { # Colors for each model
    'BERT': 'tab:blue',
    'DeBERTa': 'tab:orange',
    'RoBERTa': 'tab:green',
    'CNN': 'tab:red',
    'BiLSTM': 'tab:purple',
    'Linear': 'tab:brown',
    'NB': 'tab:pink',
    'PA': 'tab:gray'
}

MODEL_MARKERS = { # Markers for each model
    'BERT': 'o',
    'DeBERTa': 's',
    'RoBERTa': '^',
    'CNN': 'D',
    'BiLSTM': 'v',
    'Linear': 'P',
    'NB': 'X',
    'PA': '*'
}

FIG_SIZE = (10, 5) # Default figure size
plt.rcParams.update({'font.size': 12}) # Update default font size for plots


# -------------------
# Load Data Functions
# -------------------

def load_data(type='topic', dir="", alpha=None, dim=None, lam=None):
    """
    Scan the RESULTS_DIR for model folders and load their JSON results, considering only the specified type.

    Args:
        type (str): Type of results to load ('topic' or 'date').
        dir (str): Suffix for results directory.
        alpha (float, optional): Alpha value for Distillation
        dim (int, optional): Dimension for Replay
        lam (float, optional): Lambda value for EWC

    Returns:
        dict: A dictionary with model names as keys and their loaded JSON data as values.
    """

    RESULTS_DIR = f'results_{dir}'
    model_data = {}

    # -------- build suffix dynamically --------
    # order matters: alpha then dim (as in your filenames)
    suffix_parts = []

    if alpha is not None:
        suffix_parts.append(str(alpha))

    if dim is not None:
        suffix_parts.append(str(dim))

    if lam is not None:
        suffix_parts.append(str(lam))

    # build suffix string
    extra_suffix = ""
    if suffix_parts:
        extra_suffix = "_" + "_".join(suffix_parts)

    for folder_name in os.listdir(RESULTS_DIR):
        folder_path = os.path.join(RESULTS_DIR, folder_name)
        
        # Considering only folders ending with the specified type
        if os.path.isdir(folder_path) and folder_name.endswith(f'_{type}'):
            parts = folder_name.split('_')
            
            # Get MODEL name from folder "results_{MODEL}_{type}"
            model_name = parts[1] 
            
            # JSON paths for Single / Cumulative / Full
            full_path = os.path.join(folder_path, f'results_full_{type}{extra_suffix}.json')
            cumulative_path = os.path.join(folder_path, f'results_cumulative_{type}{extra_suffix}.json')
            single_path = os.path.join(folder_path, f'results_single_{type}{extra_suffix}.json')
            
            data = {'full': None, 'cumulative': None, 'single': None}
            
            # Load data from Full Topic
            if os.path.exists(full_path):
                with open(full_path, 'r') as f:
                    data['full'] = json.load(f)
            
            # Load data from Cumulative Topic
            if os.path.exists(cumulative_path):
                with open(cumulative_path, 'r') as f:
                    data['cumulative'] = json.load(f)
            
            # Load data from Single Topic
            if os.path.exists(single_path):
                with open(single_path, 'r') as f:
                    data['single'] = json.load(f)
            
            # Store only if at least one data type is present
            if data['full'] or data['cumulative'] or data['single']:
                model_data[model_name] = data
            
            # Order models data according to MODEL_ORDER
            ordered_model_data = {k: model_data[k] for k in MODEL_ORDER if k in model_data}

    return ordered_model_data


def load_offline_result(type='topic'):
    """
    Load offline (optimal) F1-score for each model.

    Args:
        type (str): 'topic' or 'date'

    Returns:
        dict: {model_name: best_f1_score}
    """

    RESULTS_DIR = "results_all_data"
    best_results = {}

    if not os.path.exists(RESULTS_DIR):
        print(f"[WARNING] Best results directory not found: {RESULTS_DIR}")
        return best_results

    for folder_name in os.listdir(RESULTS_DIR):
        folder_path = os.path.join(RESULTS_DIR, folder_name)

        # expecting: results_{MODEL}_{type}
        if os.path.isdir(folder_path) and folder_name.endswith(f"_{type}"):
            parts = folder_name.split("_")
            model_name = parts[1]

            best_path = os.path.join(folder_path, f"results_full_{type}.json")
            if os.path.exists(best_path):
                with open(best_path, "r") as f:
                    data = json.load(f)
                    best_results[model_name] = data["f1_score"]
            else:
                print(f"[WARNING] Missing results_full_{type}.json in {folder_path}")

    # order according to MODEL_ORDER
    ordered_best_results = {
        k: best_results[k] for k in MODEL_ORDER if k in best_results
    }

    return ordered_best_results


def load_random_baseline(type='topic', dir=""):
    """
    Load random baseline results (mean and std) for the given type.

    Args:
        type (str): Type of results to load ('topic' or 'date').
        dir (str): Suffix for results directory.

    Returns:
        tuple: A tuple containing mean and std of the random baseline.
    """

    RESULTS_DIR = f'results_{dir}'
    path = f"{RESULTS_DIR}/results_random_{type}/random_baseline_full_test.json"
    
    if not os.path.exists(path):
        print(f"[WARNING] Random baseline file not found: {path}")
        return None
    
    with open(path, "r") as f:
        data = json.load(f)
    
    return data["mean"], data["std"]


def load_random_baseline_by_task(type='topic', dir=""):
    """
    Load random baseline results (mean and std) for each task, given the type.

    Args:
        type (str): Type of results to load ('topic' or 'date').
        dir (str): Suffix for results directory.

    Returns:
        dict: {task: (mean, std)}
    """

    RESULTS_DIR = f'results_{dir}'
    path = f"{RESULTS_DIR}/results_random_{type}/random_baseline_by_{type}.json"
    
    if not os.path.exists(path):
        print(f"[WARNING] Random baseline file not found: {path}")
        return None
    
    with open(path, "r") as f:
        data = json.load(f)
    
    baseline_by_task = {}
    for task, stats in data.items():
        baseline_by_task[task] = (stats["mean"], stats["std"])
    
    return baseline_by_task





# ------------------------------------
# Task Matrix + ACC & BWT computations
# ------------------------------------

def build_task_matrix(single_data, task_order):
    """
    Build Task-Specific Evaluation Matrix R.

    Args:
        single_data (dict): data['single'] from load_data
        task_order (list): ordered list of tasks (topics or dates)

    Returns:
        pd.DataFrame: NxN matrix R with rows=FT tasks, columns=Test tasks
    """
    N = len(task_order)
    R = np.full((N, N), np.nan)

    for i, ft_task in enumerate(task_order):
        if ft_task in single_data:
            for j, test_task in enumerate(task_order):
                if test_task in single_data[ft_task]:
                    R[i, j] = single_data[ft_task][test_task]

    return pd.DataFrame(R, index=task_order, columns=task_order)


def compute_acc(R):
    """
    Compute final ACC (Average Accuracy).

    Args:
        R (pd.DataFrame): NxN task matrix

    Returns:
        float: ACC
    """
    last_row = R.iloc[-1]
    ACC = last_row.mean(skipna=True)
    return ACC


def compute_bwt(R):
    """
    Compute final BWT.

    Args:
        R (pd.DataFrame): NxN task matrix

    Returns:
        float: BWT
    """
    N = R.shape[0]
    diffs = []

    for i in range(N - 1):
        if not np.isnan(R.iloc[-1, i]) and not np.isnan(R.iloc[i, i]):
            diffs.append(R.iloc[-1, i] - R.iloc[i, i])

    return np.mean(diffs) if diffs else np.nan


def compute_fwt(R, baseline_by_task):
    """
    Compute Forward Transfer (FWT).

    Args:
        R (pd.DataFrame): NxN task matrix
        baseline_by_task (dict): {task: (mean,std)}

    Returns:
        float: FWT
    """
    tasks = list(R.columns)
    N = len(tasks)

    diffs = []

    for i in range(1, N):
        prev_task = tasks[i-1]
        curr_task = tasks[i]

        if curr_task not in baseline_by_task:
            continue

        baseline_mean = baseline_by_task[curr_task][0]
        val = R.iloc[i-1, i]

        if not np.isnan(val):
            diffs.append(val - baseline_mean)

    return np.mean(diffs) if diffs else np.nan



def compute_acc_progressive(R):
    """
    Compute ACC(k) for k = 1 ... N.

    Args:
        R (pd.DataFrame): NxN task matrix

    Returns:
        list: ACC values at each step k
    """
    acc_values = []

    for k in range(1, R.shape[0] + 1):
        # row k-1, columns 0..k-1
        row_k = R.iloc[k-1, :k]
        acc_k = row_k.mean(skipna=True)
        acc_values.append(acc_k)

    return acc_values


def compute_bwt_progressive(R):
    """
    Compute BWT(k) for k = 2 ... N.
    BWT(1) is undefined and set to NaN.

    Args:
        R (pd.DataFrame): NxN task matrix

    Returns:
        list: BWT values at each step k (length N)
    """
    bwt_values = [np.nan]  # BWT(1) undefined

    for k in range(2, R.shape[0] + 1):
        diffs = []
        for i in range(k - 1):
            if not np.isnan(R.iloc[k-1, i]) and not np.isnan(R.iloc[i, i]):
                diffs.append(R.iloc[k-1, i] - R.iloc[i, i])

        bwt_k = np.mean(diffs) if diffs else np.nan
        bwt_values.append(bwt_k)

    return bwt_values


def compute_fwt_progressive(R, baseline_by_task):
    """
    Progressive FWT(k)

    Args:
        R (pd.DataFrame)
        baseline_by_task (dict)

    Returns:
        list length N
    """
    tasks = list(R.columns)
    N = len(tasks)

    fwt_values = [np.nan]  # FWT(1) undefined

    for k in range(2, N + 1):
        diffs = []

        for i in range(1, k):
            curr_task = tasks[i]

            if curr_task not in baseline_by_task:
                continue

            baseline_mean = baseline_by_task[curr_task][0]
            val = R.iloc[i-1, i]

            if not np.isnan(val):
                diffs.append(val - baseline_mean)

        fwt_k = np.mean(diffs) if diffs else np.nan
        fwt_values.append(fwt_k)

    return fwt_values





# -------------------------
# Plot Functions: Full Test
# -------------------------

def plot_full_comparison(model_data, task_order, type='topic', figsize=FIG_SIZE):
    """
    Plot of F1-score on Full Test Set across all models,
    including random baseline and best (offline) results.

    Args:
        model_data (dict): Dictionary containing model results loaded from JSON files.
        task_order (list): List of topics/dates to iterate through.
        type (str): Label for the x-axis ('topic' or 'date').
        figsize (tuple): Figure size for the plot.
    """

    plt.figure(figsize=figsize)


    # --- Plot models (FULL) ---
    for model_name, data in model_data.items():
        if model_name not in MODEL_COLORS or not data['full']:
            continue

        x_values = []
        y_values = []

        for task in task_order:
            if task in data['full']:
                x_values.append(task)
                y_values.append(data['full'][task])

        plt.plot(
            x_values,
            y_values,
            color=MODEL_COLORS[model_name],
            marker=MODEL_MARKERS[model_name],
            linewidth=2,
            label=model_name
        )

    # --- Plot offline results (horizontal lines) ---
    offline_results = load_offline_result(type)

    for model_name, f1 in offline_results.items():
        # plot only models that are actually in model_data
        if model_name not in model_data:
            continue

        plt.hlines(
            y=f1,
            xmin=task_order[0],
            xmax=task_order[-1],
            colors=MODEL_COLORS.get(model_name, 'black'),
            linestyles=':',
            linewidth=2,
            label=f"{model_name} (offline)"
        )

    # --- Plot random baseline ---
    random_data = load_random_baseline(type, dir="finetuning")

    if random_data is not None:
        random_mean, random_std = random_data
        x_vals = task_order
        y_mean = [random_mean] * len(x_vals)
        y_low = [random_mean - random_std] * len(x_vals)
        y_high = [random_mean + random_std] * len(x_vals)

        plt.plot(
            x_vals,
            y_mean,
            linestyle='--',
            color='black',
            linewidth=2,
            label='Random baseline'
        )

        plt.fill_between(
            x_vals,
            y_low,
            y_high,
            color='gray',
            alpha=0.2,
            label='Random ± std'
        )

    # --- Plot settings ---
    plt.title('Model Comparison: F1-score on Full Test Set')
    plt.xlabel(f'Fine-tuning {type.capitalize()}')
    plt.ylabel('Weighted F1-score')
    plt.legend(
        title='Models',
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        borderaxespad=0.
    )
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_full_comparison_tesi(model_data, task_order, type='topic', figsize=FIG_SIZE):
    """
    Plot of F1-score on Full Test Set across all models,
    including random baseline and best (offline) results.

    Args:
        model_data (dict): Dictionary containing model results loaded from JSON files.
        task_order (list): List of topics/dates to iterate through.
        type (str): Label for the x-axis ('topic' or 'date').
        figsize (tuple): Figure size for the plot.
    """

    plt.figure(figsize=figsize)

    last_task = task_order[-1]

    # --- Plot models (FULL) ---
    for model_name, data in model_data.items():
        if model_name not in MODEL_COLORS or not data['full']:
            continue

        x_values = []
        y_values = []

        for task in task_order:
            if task in data['full']:
                x_values.append(task)
                y_values.append(data['full'][task])

        plt.plot(
            x_values,
            y_values,
            color=MODEL_COLORS[model_name],
            marker=MODEL_MARKERS[model_name],
            linewidth=2,
            label=model_name
        )

    # --- Plot offline best results (ONLY last task) ---
    offline_results = load_offline_result(type)

    for model_name, best_f1 in offline_results.items():
        if model_name not in model_data:
            continue
            
        plt.errorbar(
            last_task,
            best_f1,
            color=MODEL_COLORS[model_name],
            marker=MODEL_MARKERS[model_name],
            markersize=10,
            linestyle='None',
            label=f"{model_name} (offline)"
        )

    # --- Plot random baseline (ONLY last task) ---
    random_data = load_random_baseline(type, dir="finetuning")

    if random_data is not None:
        random_mean, random_std = random_data

        plt.errorbar(
            last_task,
            random_mean,
            yerr=random_std,
            color='black',
            marker='o',
            markersize=10,
            linestyle='None',
            capsize=5,
            label='Random (± std)'
        )

    # --- Plot settings ---
    plt.title('Model Comparison: F1-score on Full Test Set')
    plt.xlabel(f'Fine-tuning {type.capitalize()}')
    plt.ylabel('Weighted F1-score')

    # Legend outside on the right
    plt.legend(
        title='Models',
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        borderaxespad=0.
    )

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()



# -------------------------------
# Plot Functions: Cumulative Test
# -------------------------------

def plot_cumulative_comparison(model_data, task_order, type='topic', figsize=FIG_SIZE):
    """
    Plot of F1-score on Cumulative Test Set across all models.

    Args:
        model_data (dict): Dictionary containing model results loaded from JSON files.
        task_order (list): List of topics/dates to iterate through.
        type (str): Label for the x-axis ('topic' or 'date').
        figsize (tuple): Figure size for the plot.
    """

    plt.figure(figsize=figsize)

    # --- Plot models ---
    for model_name, data in model_data.items():
        if data['cumulative']:
            x_values = []
            y_values = []
            for task in task_order: # Extract values ordered according to topic/data order
                if task in data['cumulative']:
                    x_values.append(task)
                    y_values.append(data['cumulative'][task])

            plt.plot(
                x_values,
                y_values,
                color=MODEL_COLORS[model_name],
                marker=MODEL_MARKERS[model_name],
                linewidth=2,
                label=model_name
            )

    # --- Plot settings ---
    plt.title('Model Comparison: F1-score on Cumulative Test Set')
    plt.xlabel(f'Fine-tuning {type.capitalize()}')
    plt.ylabel('Weighted F1-score')
    plt.legend(title='Models', loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_cumulative_comparison_tesi(model_data, task_order, type='topic', figsize=FIG_SIZE):
    """
    Plot of F1-score on Cumulative Test Set across all models.

    Args:
        model_data (dict): Dictionary containing model results loaded from JSON files.
        task_order (list): List of topics/dates to iterate through.
        type (str): Label for the x-axis ('topic' or 'date').
        figsize (tuple): Figure size for the plot.
    """

    plt.figure(figsize=figsize)

    last_task = task_order[-1]

    # --- Plot models ---
    for model_name, data in model_data.items():
        if data['cumulative']:
            x_values = []
            y_values = []
            for task in task_order: # Extract values ordered according to topic/data order
                if task in data['cumulative']:
                    x_values.append(task)
                    y_values.append(data['cumulative'][task])

            plt.plot(
                x_values,
                y_values,
                color=MODEL_COLORS[model_name],
                marker=MODEL_MARKERS[model_name],
                linewidth=2,
                label=model_name
            )
    
    # --- Plot offline best results (ONLY last task) ---
    offline_results = load_offline_result(type)

    for model_name, best_f1 in offline_results.items():
        if model_name not in model_data:
            continue
            
        plt.errorbar(
            last_task,
            best_f1,
            color=MODEL_COLORS[model_name],
            marker=MODEL_MARKERS[model_name],
            markersize=10,
            linestyle='None',
            label=f"{model_name} (offline)"
        )

    # --- Plot random baseline (ONLY last task) ---
    random_data = load_random_baseline(type)

    if random_data is not None:
        random_mean, random_std = random_data

        plt.errorbar(
            last_task,
            random_mean,
            yerr=random_std,
            color='black',
            marker='o',
            markersize=10,
            linestyle='None',
            capsize=5,
            label='Random (± std)'
        )

    # --- Plot settings ---
    plt.title('Model Comparison: F1-score on Cumulative Test Set')
    plt.xlabel(f'Fine-tuning {type.capitalize()}')
    plt.ylabel('Weighted F1-score')

    # Legend outside on the right
    plt.legend(
        title='Models',
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        borderaxespad=0.
    )

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()



# ----------------------------
# Plot Functions: Single Tests
# ----------------------------

def plot_single_comparisons(model_data, task_order, type='topic', figsize=FIG_SIZE):
    """
    Plot of F1-score on Single Topic Test Sets across all models arranged in a 2-column grid.

    Args:
        model_data (dict): Dictionary containing model results loaded from JSON files.
        task_order (list): List of topics/dates to iterate through.
        type (str): Label for the x-axis ('topic' or 'date').
        figsize (tuple): Figure size for the plot.
    """

    random_baseline = load_random_baseline_by_task(type, dir="finetuning")
    
    # Number of columns and rows needed for subplots
    num_plots = len(task_order)
    num_cols = 2
    num_rows = math.ceil(num_plots / num_cols)
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(figsize[0]+5, figsize[1] * num_rows))
    axes = axes.flatten()

    # Iterate over each topic/date and the corresponding axis
    for i, target_test_type in enumerate(task_order):
        ax = axes[i] # Select the subplot axis
        has_data = False
        
        for model_name, data in model_data.items():
            if data['single']:
                y_values = []
                x_values = []
                
                # Get scores for the target_test_type across all fine-tuning topics/dates
                for ft_type in task_order:
                    # JSON structure: {FT_Type: {Test_Type: score, ...}}
                    if ft_type in data['single']:
                        scores_dict = data['single'][ft_type]
                        if target_test_type in scores_dict:
                            x_values.append(ft_type)
                            y_values.append(scores_dict[target_test_type])
                if y_values:
                    ax.plot(
                        x_values, 
                        y_values,
                        color=MODEL_COLORS[model_name],
                        marker=MODEL_MARKERS[model_name],
                        label=model_name)
                    has_data = True

        # add random baseline if available
        if target_test_type in random_baseline:
            mean, std = random_baseline[target_test_type]
            ax.errorbar(
                [target_test_type],
                [mean],
                yerr=[std],
                color='black',
                marker='o',
                markersize=8,
                linestyle='None',
                capsize=5,
                label='Random (± std)'
            )
        
        # Graphic settings for the single subplot
        if has_data:
            ax.set_title(f'Test Set: "{target_test_type.upper()}"')
            ax.set_xlabel(f'Fine-tuning {type.capitalize()}')
            ax.set_ylabel('F1-score')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(
                fontsize='small',
                title='Models',
                bbox_to_anchor=(1.02, 1),
                loc='upper left',
                borderaxespad=0.
            )
        else:
            ax.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center')

    # Remove empty axes (if the number of plots is odd)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_single_comparisons_start_task(model_data, task_order, type='topic', figsize=FIG_SIZE):
    """
    Plot of F1-score on Single Topic Test Sets across all models arranged in a 2-column grid.

    Args:
        model_data (dict): Dictionary containing model results loaded from JSON files.
        task_order (list): List of topics/dates to iterate through.
        type (str): Label for the x-axis ('topic' or 'date').
        figsize (tuple): Figure size for the plot.
    """

    random_baseline = load_random_baseline_by_task(type, dir="finetuning")
    last_task = task_order[-1]

    # Number of columns and rows needed for subplots
    num_plots = len(task_order)
    num_cols = 2
    num_rows = math.ceil(num_plots / num_cols)
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(figsize[0]+5, figsize[1] * num_rows))
    axes = axes.flatten()

    # Iterate over each topic/date and the corresponding axis
    for i, target_test_type in enumerate(task_order):
        ax = axes[i]
        has_data = False

        for model_name, data in model_data.items():
            if data['single']:
                y_values = []
                x_values = []

                # 🔑 ONLY from the current topic onward
                for ft_type in task_order[i:]:
                    if ft_type in data['single']:
                        scores_dict = data['single'][ft_type]
                        if target_test_type in scores_dict:
                            x_values.append(ft_type)
                            y_values.append(scores_dict[target_test_type])

                if y_values:
                    ax.plot(
                        x_values, 
                        y_values,
                        color=MODEL_COLORS[model_name],
                        marker=MODEL_MARKERS[model_name],
                        label=model_name)
                    has_data = True
        
        # add random baseline if available
        if target_test_type in random_baseline:
            mean, std = random_baseline[target_test_type]
            ax.errorbar(
                last_task,
                [mean],
                yerr=[std],
                color='black',
                marker='o',
                markersize=8,
                linestyle='None',
                capsize=5,
                label='Random (± std)'
            )

        # Graphic settings for the single subplot
        if has_data:
            ax.set_title(f'Test Set: "{target_test_type.upper()}"')
            ax.set_xlabel(f'Fine-tuning {type.capitalize()}')
            ax.set_ylabel('F1-score')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(
                fontsize='small',
                title='Models',
                bbox_to_anchor=(1.02, 1),
                loc='upper left',
                borderaxespad=0.
            )
        else:
            ax.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center')

    # Remove empty axes (if the number of plots is odd)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()





# ------------------------------------------
# Plot Functions: Heatmaps + ACC & BWT plots
# ------------------------------------------

def plot_task_matrices(model_data, task_order, type='topic', figsize=FIG_SIZE):
    """
    Plot heatmaps of Task-Specific Evaluation Matrix R for each model.

    Args:
        model_data (dict): Dictionary containing model results loaded from JSON files.
        task_order (list): List of topics/dates to iterate through.
        type (str): Label for the title ('topic' or 'date').
        figsize (tuple): Figure size for the plot.
    """

    models = list(model_data.keys())
    num_models = len(models)

    num_cols = 3
    num_rows = math.ceil(num_models / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(figsize[0]+5, figsize[1] * num_rows))
    axes = axes.flatten()

    # Compute global min/max for consistent color scale
    all_values = []
    for model in models:
        single_data = model_data[model]['single']
        if single_data:
            R = build_task_matrix(single_data, task_order)
            all_values.extend(R.values.flatten())

    vmin = np.nanmin(all_values)
    vmax = np.nanmax(all_values)

    for i, model_name in enumerate(models):
        ax = axes[i]
        single_data = model_data[model_name]['single']

        if not single_data:
            ax.set_title(f"{model_name} (No Data)")
            ax.axis('off')
            continue

        R = build_task_matrix(single_data, task_order)

        sns.heatmap(
            R,
            ax=ax,
            cmap='viridis',
            vmin=vmin,
            vmax=vmax,
            annot=True,
            fmt=".2f",
            cbar=True # colorbar on everyone
        )

        ax.set_title(model_name)
        ax.set_xlabel("Test Task")
        ax.set_ylabel("Fine-tuning Task")

        # rotate task labels to 45°
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # remove empty axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f"Task-Specific Evaluation Matrix ({type.capitalize()})", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def plot_acc_comparison(model_data, task_order, type='topic', figsize=FIG_SIZE):
    """
    Bar plot comparing ACC across models.
    """

    models = []
    accs = []
    colors = []

    for model_name, data in model_data.items():
        if not data['single']:
            continue

        R = build_task_matrix(data['single'], task_order)
        ACC = compute_acc(R)

        models.append(model_name)
        accs.append(ACC)
        colors.append(MODEL_COLORS.get(model_name, 'black'))

    plt.figure(figsize=figsize)
    plt.bar(models, accs, color=colors)

    plt.title("Average Accuracy (ACC) Comparison")
    plt.xlabel("Model")
    plt.ylabel("ACC (Weighted F1)")
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_bwt_comparison(model_data, task_order, type='topic', figsize=FIG_SIZE):
    """
    Bar plot comparing BWT across models.

    Args:
        model_data (dict): Dictionary containing model results loaded from JSON files.
        task_order (list): List of topics/dates to iterate through.
        type (str): Label for the title ('topic' or 'date').
        figsize (tuple): Figure size for the plot.
    """

    models = []
    bwts = []
    colors = []

    for model_name, data in model_data.items():
        if not data['single']:
            continue

        R = build_task_matrix(data['single'], task_order)
        BWT = compute_bwt(R)

        models.append(model_name)
        bwts.append(BWT)
        colors.append(MODEL_COLORS.get(model_name, 'black'))

    plt.figure(figsize=figsize)
    plt.bar(models, bwts, color=colors)
    plt.axhline(0, linestyle='--', color='black', linewidth=1)

    plt.title("Backward Transfer (BWT) Comparison")
    plt.xlabel("Model")
    plt.ylabel("BWT")
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_fwt_comparison(model_data, task_order, type='topic', dir="finetuning", figsize=FIG_SIZE):
    """
    Bar plot comparing FWT across models.
    """

    baseline_by_task = load_random_baseline_by_task(type=type, dir=dir)
    if baseline_by_task is None:
        print("Random baseline not found. Cannot compute FWT.")
        return

    models = []
    fwts = []
    colors = []

    for model_name, data in model_data.items():
        if not data['single']:
            continue

        R = build_task_matrix(data['single'], task_order)
        FWT = compute_fwt(R, baseline_by_task)

        models.append(model_name)
        fwts.append(FWT)
        colors.append(MODEL_COLORS.get(model_name, 'black'))

    plt.figure(figsize=figsize)
    plt.bar(models, fwts, color=colors)
    plt.axhline(0, linestyle='--', color='black', linewidth=1)

    plt.title("Forward Transfer (FWT) Comparison")
    plt.xlabel("Model")
    plt.ylabel("FWT")
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()



def plot_acc_progressive(model_data, task_order, type='topic', figsize=FIG_SIZE):
    """
    Plot ACC(k) over tasks for all models.

    Args:
        model_data (dict): Dictionary containing model results loaded from JSON files.
        task_order (list): List of topics/dates to iterate through.
        type (str): Label for the title ('topic' or 'date').
        figsize (tuple): Figure size for the plot.
    """

    plt.figure(figsize=figsize)

    for model_name, data in model_data.items():
        if not data['single']:
            continue

        R = build_task_matrix(data['single'], task_order)
        acc_values = compute_acc_progressive(R)

        plt.plot(
            task_order,
            acc_values,
            color=MODEL_COLORS[model_name],
            marker=MODEL_MARKERS[model_name],
            linewidth=2,
            label=model_name
        )

    plt.title("Progressive Average Accuracy (ACC)")
    plt.xlabel(f"Training {type.capitalize()}")
    plt.ylabel("ACC (Weighted F1)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Models", loc='best')
    plt.tight_layout()
    plt.show()


def plot_bwt_progressive(model_data, task_order, type='topic', figsize=FIG_SIZE):
    """
    Plot BWT(k) over tasks for all models.

    Args:
        model_data (dict): Dictionary containing model results loaded from JSON files.
        task_order (list): List of topics/dates to iterate through.
        type (str): Label for the title ('topic' or 'date').
        figsize (tuple): Figure size for the plot.
    """

    plt.figure(figsize=figsize)

    for model_name, data in model_data.items():
        if not data['single']:
            continue

        R = build_task_matrix(data['single'], task_order)
        bwt_values = compute_bwt_progressive(R)

        plt.plot(
            task_order,
            bwt_values,
            color=MODEL_COLORS[model_name],
            marker=MODEL_MARKERS[model_name],
            linewidth=2,
            label=model_name
        )

    plt.axhline(0, linestyle='--', color='black', linewidth=1)

    plt.title("Progressive Backward Transfer (BWT)")
    plt.xlabel(f"Training {type.capitalize()}")
    plt.ylabel("BWT")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Models", loc='best')
    plt.tight_layout()
    plt.show()


def plot_fwt_progressive(model_data, task_order, type='topic', dir="finetuning", figsize=FIG_SIZE):
    """
    Plot FWT(k) over tasks for all models.
    """

    baseline_by_task = load_random_baseline_by_task(type=type, dir=dir)
    if baseline_by_task is None:
        print("Random baseline not found. Cannot compute FWT.")
        return

    plt.figure(figsize=figsize)

    for model_name, data in model_data.items():
        if not data['single']:
            continue

        R = build_task_matrix(data['single'], task_order)
        fwt_values = compute_fwt_progressive(R, baseline_by_task)

        plt.plot(
            task_order,
            fwt_values,
            color=MODEL_COLORS[model_name],
            marker=MODEL_MARKERS[model_name],
            linewidth=2,
            label=model_name
        )

    plt.axhline(0, linestyle='--', color='black', linewidth=1)

    plt.title("Progressive Forward Transfer (FWT)")
    plt.xlabel(f"Training {type.capitalize()}")
    plt.ylabel("FWT")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Models", loc='best')
    plt.tight_layout()
    plt.show()
