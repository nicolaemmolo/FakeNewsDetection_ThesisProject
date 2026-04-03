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

# Technique order for plots
TECHNIQUE_ORDER = [
    'Finetuning',
    'Replay (n=50)',
    'Replay (n=100)',
    'Replay (n=200)',
    'LwF (α=03)',
    'LwF (α=05)',
    'EWC (λ=300)',
    'EWC (λ=500)',
    'EWC (λ=1000)',
    'Hybrid (α=03, n=50)',
    'Hybrid (α=03, n=100)',
    'Hybrid (α=03, n=200)',
    'Hybrid (α=05, n=50)',
    'Hybrid (α=05, n=100)',
    'Hybrid (α=05, n=200)'
]

# Colors for each technique (different shades for same techniques)
TECHNIQUE_COLORS = { 
    'Finetuning': 'tab:blue',
    'Replay (n=50)': 'tab:orange',
    'Replay (n=100)': 'tab:orange',
    'Replay (n=200)': 'tab:orange',
    'LwF (α=03)': 'tab:green',
    'LwF (α=05)': 'tab:green',
    'EWC (λ=300)': 'tab:purple',
    'EWC (λ=500)': 'tab:purple',
    'EWC (λ=1000)': 'tab:purple',
    'Hybrid (α=03, n=50)': 'tab:brown',
    'Hybrid (α=03, n=100)': 'tab:brown',
    'Hybrid (α=03, n=200)': 'tab:brown',
    'Hybrid (α=05, n=50)': 'tab:pink',
    'Hybrid (α=05, n=100)': 'tab:pink',
    'Hybrid (α=05, n=200)': 'tab:pink'
}

# Markers for each technique
TECHNIQUE_MARKERS = { 
    'Finetuning': 'o',
    'Replay (n=50)': 'D',
    'Replay (n=100)': 'P',
    'Replay (n=200)': 'X',
    'LwF (α=03)': 'D',
    'LwF (α=05)': 'P',
    'EWC (λ=300)': 'D',
    'EWC (λ=500)': 'P',
    'EWC (λ=1000)': 'X',
    'Hybrid (α=03, n=50)': 'D',
    'Hybrid (α=03, n=100)': 'P',
    'Hybrid (α=03, n=200)': 'X',
    'Hybrid (α=05, n=50)': 's',
    'Hybrid (α=05, n=100)': '*',
    'Hybrid (α=05, n=200)': 'v'
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


def load_data_for_model_comparison(model_target, type='topic'):
    """
    Carica tutti i risultati di tutte le tecniche per un singolo modello specifico.
    """
    all_results = {}

    # 1. Finetuning (Senza parametri)
    data = load_data(type=type, dir="finetuning")
    if model_target in data:
        all_results["Finetuning"] = data[model_target]

    # 2. LwF (alpha)
    for a in ["03", "05"]:
        data = load_data(type=type, dir="distillation", alpha=a)
        if model_target in data:
            all_results[f"LwF (α={a})"] = data[model_target]

    # 3. Replay (dim)
    for d in [50, 100, 200]:
        data = load_data(type=type, dir="replay", dim=d)
        if model_target in data:
            all_results[f"Replay (n={d})"] = data[model_target]

    # 4. EWC (lambda)
    for l in ["300", "500", "1000"]:
        data = load_data(type=type, dir="ewc", lam=l)
        if model_target in data:
            all_results[f"EWC (λ={l})"] = data[model_target]

    # 5. Hybrid (alpha & dim)
    hybrid_params = [("03", 50), ("03", 100), ("03", 200), ("05", 50), ("05", 100), ("05", 200)]
    for a, d in hybrid_params:
        data = load_data(type=type, dir="dist_rep", alpha=a, dim=d)
        if model_target in data:
            all_results[f"Hybrid (α={a}, n={d})"] = data[model_target]

    return all_results





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





# --------------
# Plot Functions
# --------------

def plot_full_comparison_model(model_name, tech_data, task_order, type='topic', figsize=FIG_SIZE):
    """
    Versione 'Tesi': Confronto F1-score su Full Test Set per un singolo modello.
    Mostra l'evoluzione delle tecniche e i baseline (offline/random) solo sull'ultimo task.
    """
    plt.figure(figsize=figsize)
    
    last_task = task_order[-1]

    # --- Plot delle Tecniche (Linee evolutive) ---
    # Usiamo TECHNIQUE_ORDER per assicurarci che l'ordine in legenda sia quello desiderato
    for tech_name in TECHNIQUE_ORDER:
        if tech_name in tech_data:
            data = tech_data[tech_name]
            if not data['full']:
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
                color=TECHNIQUE_COLORS.get(tech_name, 'gray'),
                marker=TECHNIQUE_MARKERS.get(tech_name, 'o'),
                linewidth=2,
                label=tech_name
            )

    # --- Plot Random Baseline (Punto isolato con barra di errore) ---
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

    # --- Impostazioni Grafiche ---
    plt.title(f'Model: {model_name} - CL Techniques Comparison (Full Test Set)')
    plt.xlabel(f'Fine-tuning {type.capitalize()}')
    plt.ylabel('Weighted F1-score')

    # Posizionamento legenda a destra
    plt.legend(
        title='Techniques',
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        borderaxespad=0.
    )

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()



def plot_cumulative_comparison_model(model_name, tech_data, task_order, type='topic', figsize=FIG_SIZE):
    """
    Versione 'Tesi': Confronto F1-score su Cumulative Test Set per un singolo modello.
    Mostra come la performance media sui task visti finora cambia col tempo.
    """
    plt.figure(figsize=figsize)

    last_task = task_order[-1]

    # --- Plot delle Tecniche (Dati Cumulativi) ---
    # Usiamo TECHNIQUE_ORDER per mantenere la coerenza visiva in legenda
    for tech_name in TECHNIQUE_ORDER:
        if tech_name in tech_data:
            data = tech_data[tech_name]
            # Verifichiamo che esistano i dati cumulativi
            if data['cumulative']:
                x_values = []
                y_values = []
                
                for task in task_order:
                    if task in data['cumulative']:
                        x_values.append(task)
                        y_values.append(data['cumulative'][task])

                if x_values:
                    plt.plot(
                        x_values,
                        y_values,
                        color=TECHNIQUE_COLORS.get(tech_name, 'gray'),
                        marker=TECHNIQUE_MARKERS.get(tech_name, 'o'),
                        linewidth=2,
                        label=tech_name
                    )

    # --- Plot Random Baseline (Sull'ultimo task) ---
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

    # --- Impostazioni Grafiche ---
    plt.title(f'Model: {model_name} - CL Techniques Comparison (Cumulative Test Set)')
    plt.xlabel(f'Fine-tuning {type.capitalize()}')
    plt.ylabel('Weighted F1-score')

    # Posizionamento legenda a destra per ospitare tutte le combinazioni
    plt.legend(
        title='Techniques',
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        borderaxespad=0.
    )

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()



def plot_single_comparisons_model_start_task(model_name, tech_data, task_order, type='topic', figsize=FIG_SIZE):
    """
    Griglia di plot (uno per ogni task). Mostra come le diverse tecniche (per un singolo modello)
    performano su quel task specifico durante il ciclo di training.
    """
    random_baseline = load_random_baseline_by_task(type, dir="finetuning")
    last_task = task_order[-1]

    # Calcolo dimensioni griglia
    num_plots = len(task_order)
    num_cols = 2
    num_rows = math.ceil(num_plots / num_cols)
    
    # Aumentiamo un po' la larghezza della figura per far stare la legenda tecnica
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(figsize[0]+6, figsize[1] * num_rows))
    axes = axes.flatten()

    # Ciclo sui TEST SET (i quadranti della griglia)
    for i, target_test_type in enumerate(task_order):
        ax = axes[i]
        has_data = False
        
        # Ciclo sulle TECNICHE (le linee nel plot) nell'ordine specificato
        for tech_name in TECHNIQUE_ORDER:
            if tech_name in tech_data:
                data = tech_data[tech_name]
                
                if data['single']:
                    y_values = []
                    x_values = []
                    
                    # 🔑 Logica start_task: analizziamo il comportamento dal task i in poi
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
                            color=TECHNIQUE_COLORS.get(tech_name, 'gray'),
                            marker=TECHNIQUE_MARKERS.get(tech_name, 'o'),
                            label=tech_name,
                            linewidth=1.5,
                            markersize=5
                        )
                        has_data = True

        # Aggiunta Random Baseline specifica per il task nel quadrante
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
        
        # Formattazione del subplot
        if has_data:
            ax.set_title(f'Model: {model_name} | Test Set: "{target_test_type.upper()}"')
            ax.set_xlabel(f'Fine-tuning Step')
            ax.set_ylabel('F1-score')
            ax.grid(True, linestyle='--', alpha=0.5)
            # Legenda solo sul lato destro per ogni subplot (o puoi metterla globale)
            ax.legend(
                fontsize='x-small',
                title='Techniques',
                bbox_to_anchor=(1.02, 1),
                loc='upper left',
                borderaxespad=0.
            )
        else:
            ax.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center')

    # Rimuovi assi vuoti se presenti
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()




def plot_task_matrices_by_technique(model_name, tech_data, task_order, type='topic', figsize=FIG_SIZE):
    """
    Genera una griglia di heatmap, una per ogni tecnica di Continual Learning,
    mostrando la matrice di valutazione R per un singolo modello.
    """
    # Filtriamo solo le tecniche che hanno effettivamente dati presenti
    available_techniques = [t for t in TECHNIQUE_ORDER if t in tech_data and tech_data[t]['single']]
    num_techs = len(available_techniques)

    if num_techs == 0:
        print(f"Nessun dato 'single' trovato per il modello {model_name}")
        return

    # Calcolo dimensioni griglia (3 colonne è di solito un buon compromesso)
    if num_techs == 4:
        num_cols = 2
    else:
        num_cols = 3
    num_rows = math.ceil(num_techs / num_cols)

    if num_cols == 2:
        dimX = figsize[0] + 2
    else:
        dimX = figsize[0] + 7
    dimY = figsize[1] * num_rows
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(dimX, dimY))
    axes = axes.flatten()

    # 1. Calcolo globale di min/max per avere una scala di colori uniforme
    all_values = []
    matrices = {}
    
    for tech in available_techniques:
        single_data = tech_data[tech]['single']
        R = build_task_matrix(single_data, task_order)
        matrices[tech] = R
        all_values.extend(R.values.flatten())

    vmin = np.nanmin(all_values)
    vmax = np.nanmax(all_values)

    # 2. Generazione delle Heatmap
    for i, tech_name in enumerate(available_techniques):
        ax = axes[i]
        R = matrices[tech_name]

        sns.heatmap(
            R,
            ax=ax,
            cmap='viridis',
            vmin=vmin,
            vmax=vmax,
            annot=True,
            fmt=".2f",
            cbar=True
        )

        ax.set_title(f"Tech: {tech_name}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Test Task")
        ax.set_ylabel("Fine-tuning Task")

        # Ruota le etichette per leggibilità
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # Rimuovi eventuali assi vuoti
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f"Task-Specific Evaluation Matrices - Model: {model_name} ({type.capitalize()})", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()



def plot_acc_progressive_model(model_name, tech_data, task_order, type='topic', figsize=FIG_SIZE):
    """
    Plot di ACC(k) per confrontare le tecniche su un singolo modello.
    """
    plt.figure(figsize=figsize)

    for tech_name in TECHNIQUE_ORDER:
        if tech_name in tech_data and tech_data[tech_name]['single']:
            data = tech_data[tech_name]
            R = build_task_matrix(data['single'], task_order)
            acc_values = compute_acc_progressive(R)

            plt.plot(
                task_order,
                acc_values,
                color=TECHNIQUE_COLORS.get(tech_name, 'gray'),
                marker=TECHNIQUE_MARKERS.get(tech_name, 'o'),
                linewidth=2,
                label=tech_name
            )

    plt.title(f"Progressive Average Accuracy (ACC) - Model: {model_name}")
    plt.xlabel(f"Training {type.capitalize()}")
    plt.ylabel("ACC (Weighted F1)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Techniques", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_bwt_progressive_model(model_name, tech_data, task_order, type='topic', figsize=FIG_SIZE):
    """
    Plot di BWT(k) per confrontare le tecniche su un singolo modello.
    """
    plt.figure(figsize=figsize)

    for tech_name in TECHNIQUE_ORDER:
        if tech_name in tech_data and tech_data[tech_name]['single']:
            data = tech_data[tech_name]
            R = build_task_matrix(data['single'], task_order)
            bwt_values = compute_bwt_progressive(R)

            plt.plot(
                task_order,
                bwt_values,
                color=TECHNIQUE_COLORS.get(tech_name, 'gray'),
                marker=TECHNIQUE_MARKERS.get(tech_name, 'o'),
                linewidth=2,
                label=tech_name
            )

    plt.axhline(0, linestyle='--', color='black', linewidth=1)
    plt.title(f"Progressive Backward Transfer (BWT) - Model: {model_name}")
    plt.xlabel(f"Training {type.capitalize()}")
    plt.ylabel("BWT")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Techniques", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_fwt_progressive_model(model_name, tech_data, task_order, type='topic', dir="finetuning", figsize=FIG_SIZE):
    """
    Plot di FWT(k) per confrontare le tecniche su un singolo modello.
    """
    baseline_by_task = load_random_baseline_by_task(type=type, dir=dir)
    if baseline_by_task is None:
        print("Random baseline not found. Cannot compute FWT.")
        return

    plt.figure(figsize=figsize)

    for tech_name in TECHNIQUE_ORDER:
        if tech_name in tech_data and tech_data[tech_name]['single']:
            data = tech_data[tech_name]
            R = build_task_matrix(data['single'], task_order)
            fwt_values = compute_fwt_progressive(R, baseline_by_task)

            plt.plot(
                task_order,
                fwt_values,
                color=TECHNIQUE_COLORS.get(tech_name, 'gray'),
                marker=TECHNIQUE_MARKERS.get(tech_name, 'o'),
                linewidth=2,
                label=tech_name
            )

    plt.axhline(0, linestyle='--', color='black', linewidth=1)
    plt.title(f"Progressive Forward Transfer (FWT) - Model: {model_name}")
    plt.xlabel(f"Training {type.capitalize()}")
    plt.ylabel("FWT")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Techniques", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.show()