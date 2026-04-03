import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re



# -------------------------
# Text statistics functions
# -------------------------

def count_labels(df):
    """
    Count total number of samples and number of samples per label in the given dataframe.
    
    Args:
        df (pd.DataFrame): DataFrame containing a 'labels' column.
    """

    # print total length
    print("Total number of rows:", len(df))
    # print number of 0 labels
    print("Number of Real News:", (df['labels'] == 0).sum())
    # print number of 1 labels
    print("Number of Fake News:", (df['labels'] == 1).sum())


def compute_text_statistics(df):
    s = df["texts"].fillna("")

    char_count = s.str.len() # number of characters

    word_count = s.str.split().str.len() # number of words

    sentence_count = s.str.count(r"[.!?]") + 1 # number of sentences (split by ., !, ?)

    uppercase_word_count = s.str.findall(r"\b[A-Z]{2,}\b").str.len() # number of uppercase words (at least 2 letters)

    punct_count = s.str.count(r"[!?]") # number of exclamation and question marks

    return {
        "avg_characters": round(char_count.mean()),
        "avg_words": round(word_count.mean()),
        "avg_sentences": round(sentence_count.mean()),
        "avg_uppercase_words": round(uppercase_word_count.mean()),
        "avg_excl_interr_points": round(punct_count.mean()),
    }


def get_frequent_tokens(df, min_frequency=10):
    """
    Get a DataFrame of tokens and their frequencies from the 'texts' column of the input DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing a 'texts' column.
        min_frequency (int): Minimum frequency threshold for tokens to be included.

    Returns:
        pd.DataFrame: A DataFrame with columns 'token' and 'frequency', filtered by min_frequency.
    """

    texts = df["texts"].fillna("").tolist() # text to list

    vectorizer = CountVectorizer( # vectorizer for efficient token counting
        lowercase=True,
        stop_words="english",
        token_pattern=r"(?u)\b\w+\b"
    )

    X = vectorizer.fit_transform(texts) # sparse matrix of token counts

    freqs = X.sum(axis=0).A1 # sum frequencies for each token

    # build dataframe token → frequency
    tokens = vectorizer.get_feature_names_out()
    freq_df = pd.DataFrame({"token": tokens, "frequency": freqs})

    freq_df = freq_df[freq_df["token"].str.len() > 1] # remove single-character tokens

    # filter by threshold and sort by frequency
    freq_df = freq_df[freq_df["frequency"] >= min_frequency]
    freq_df = freq_df.sort_values("frequency", ascending=False).reset_index(drop=True)

    return freq_df


def print_stats(df, type="REAL"):
    """
    Print text statistics and frequent tokens for the given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing news articles.
        type (str): Type of news ("REAL" or "FAKE") for display purposes.
    """

    stats = compute_text_statistics(df)

    token_freqs = get_frequent_tokens(df, min_frequency=20)

    print(f"\n\n--- Statistics for {type} news ---\n")
    print("Average characters per article:", stats["avg_characters"])
    print("Average words per article:", stats["avg_words"])
    print("Average sentences per article:", stats["avg_sentences"])
    print("Average uppercase words per article:", stats["avg_uppercase_words"])
    print("Average exclamation and question marks per article:", stats["avg_excl_interr_points"])

    print("\nMost frequent tokens:")
    print(token_freqs.head(20))



# -------------------------
# Plotting functions
# -------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import math
from sklearn.feature_extraction.text import CountVectorizer

def compute_features(df):
    """Calcola le feature statistiche per ogni testo nel dataframe."""
    df = df.copy()
    
    # 1. Numero di caratteri
    df['char_count'] = df['texts'].str.len()
    
    # 2. Numero di parole
    df['word_count'] = df['texts'].apply(lambda x: len(str(x).split()))
    
    # 3. Numero di frasi (split per . ! ?)
    df['sentence_count'] = df['texts'].apply(lambda x: len(re.split(r'[.!?]+', str(x))))
    
    # 4. Parole in maiuscolo (almeno 2 lettere)
    df['uppercase_count'] = df['texts'].apply(lambda x: len([w for w in str(x).split() if len(w) >= 2 and w.isupper()]))
    
    # 5. Punti esclamativi e interrogativi
    df['punct_count'] = df['texts'].apply(lambda x: len(re.findall(r'[!?]', str(x))))
    
    return df

def generate_thesis_plots(dataset_dict, fig_size=(14, 8), font_size=15):
    """Genera i plot richiesti per la tesi."""
    
    # Impostazioni stile
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': font_size})
    
    # Pre-processamento: uniamo tutto in un unico DF lungo per i plot comparativi
    all_data = []
    for name, df in dataset_dict.items():
        processed_df = compute_features(df)
        processed_df['dataset_name'] = name
        all_data.append(processed_df)
    
    full_df = pd.concat(all_data, ignore_index=True)
    full_df['label_str'] = full_df['labels'].map({0: 'Real', 1: 'Fake'})

    # --- PLOT 1: Real vs Fake Affiancati ---
    plt.figure(figsize=fig_size)
    # '#4daf4a' for Real '#e41a1c' for Fake
    sns.countplot(data=full_df, x='dataset_name', hue='label_str', palette={'Real': '#4daf4a', 'Fake': '#e41a1c'})
    plt.title("Datasets Distribution - Real vs Fake (Side-by-Side)", fontsize=font_size+6)
    plt.ylabel("Sample Count", fontsize=font_size+4)
    plt.xlabel("Data", fontsize=font_size+4)
    plt.xticks(rotation=45, fontsize=font_size+2)
    plt.yticks(fontsize=font_size+2)
    plt.yscale('log')
    plt.legend(title="Class", loc='upper right')
    plt.tight_layout()
    plt.show()

    # --- PLOT 2: Boxplots e Barplots per Feature Statistiche ---
    features_to_plot = [
        ('char_count', 'Number of Characters'),
        ('word_count', 'Number of Words'),
        ('sentence_count', 'Number of Sentences'),
        ('uppercase_count', 'Uppercase Words'),
        ('punct_count', 'Exclamation/Question Marks')
    ]

    # Boxplots delle distribuzioni
    for feat, label in features_to_plot:
        plt.figure(figsize=fig_size)
        sns.boxplot(data=full_df,
                    x='dataset_name',
                    y=feat,
                    hue='label_str', 
                    palette={'Real': '#4daf4a', 'Fake': '#e41a1c'},
                    showfliers=True
        )
        plt.title(f"Distribution: {label}", fontsize=font_size+5)
        plt.ylabel(label)
        plt.xlabel("Data")
        plt.xticks(rotation=45)
        plt.yscale('log')  # Scala logaritmica utile se i testi variano molto (es. Twitter vs Articoli)
        plt.legend(title="Class", loc='upper right')
        plt.tight_layout()
        plt.show()

    # Barplots delle medie
    for feat, label in features_to_plot:
        plt.figure(figsize=fig_size)
        # sns.barplot calcola automaticamente la MEDIA (mean) per ogni gruppo
        # estimator=np.mean è il default. 
        # errorbar=None rimuove la linea del calcolo dell'intervallo di confidenza
        sns.barplot(
            data=full_df, 
            x='dataset_name', 
            y=feat, 
            hue='label_str', 
            palette={'Real': '#4daf4a', 'Fake': '#e41a1c'},
            errorbar=None 
        )
        plt.title(f"Comparison of Means: {label}", fontsize=font_size+5)
        plt.ylabel("Average Value")
        plt.xlabel("Data")
        plt.xticks(rotation=45)
        # Manteniamo la scala logaritmica se i dataset sono molto diversi tra loro
        # Se i valori sono simili, puoi commentare la riga sotto
        plt.yscale('log')
        plt.legend(title="Class", loc='upper right')
        plt.tight_layout()
        plt.show()


    # PLOT 4: Heatmap di Similarità Lessicale (Jaccard)
    print("Calcolo Similarità Jaccard in corso...")
    
    # 1. Estrazione Vocabolari
    topics = list(dataset_dict.keys())
    vocabularies = {}
    
    # Limitiamo a 5000 feature per velocità e per rimuovere rumore (parole troppo rare)
    vectorizer = CountVectorizer(stop_words='english', max_features=5000, token_pattern=r"(?u)\b\w\w+\b")
    
    for name, df in dataset_dict.items():
        text_data = df['texts'].fillna("")
        try:
            vectorizer.fit(text_data)
            vocabularies[name] = set(vectorizer.get_feature_names_out())
        except ValueError:
            vocabularies[name] = set() # Gestione casi vuoti

    # 2. Calcolo Matrice
    n_topics = len(topics)
    jaccard_matrix = np.zeros((n_topics, n_topics))

    for i in range(n_topics):
        for j in range(n_topics):
            name_a = topics[i]
            name_b = topics[j]
            set_a = vocabularies[name_a]
            set_b = vocabularies[name_b]
            
            if len(set_a) == 0 or len(set_b) == 0:
                score = 0
            else:
                intersection = len(set_a.intersection(set_b))
                union = len(set_a.union(set_b))
                score = intersection / union
            jaccard_matrix[i, j] = score

    # 3. Plotting Heatmap
    plt.figure(figsize=(10, 8)) # Quadrata per la matrice
    mask = np.triu(np.ones_like(jaccard_matrix, dtype=bool), k=1) # Maschera triangolo superiore
    
    sns.heatmap(
        jaccard_matrix, 
        xticklabels=topics, 
        yticklabels=topics,
        annot=True, 
        fmt=".2f", 
        cmap="YlGnBu", 
        mask=mask,
        square=True
    )
    plt.title("Analysis of Lexical Similarity (Jaccard Index)", fontsize=font_size+6, weight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=font_size+2)
    plt.yticks(rotation=0, fontsize=font_size+2)
    # change font size to cbar

    plt.tight_layout()
    plt.show()



    # PLOT 5: Top Tokens Comparison (Bar Charts Orizzontali)
    print("Generazione Top Tokens Charts...")
    
    n_datasets = len(dataset_dict)
    cols = 3 # Numero di colonne nella griglia
    rows = math.ceil(n_datasets / cols)
    
    # Adattiamo l'altezza della figura in base al numero di righe
    fig_h = rows * 4 
    fig, axes = plt.subplots(rows, cols, figsize=(18, fig_h))
    axes = axes.flatten() # Appiattiamo per iterare facilmente

    # Ri-inizializziamo vectorizer per conteggi completi
    count_vec = CountVectorizer(stop_words='english', max_features=5000)

    for i, (name, df) in enumerate(dataset_dict.items()):
        ax = axes[i]
        text_data = df['texts'].fillna("")
        
        try:
            X = count_vec.fit_transform(text_data)
            counts = np.asarray(X.sum(axis=0)).flatten()
            words = count_vec.get_feature_names_out()
            
            # Creiamo DF temporaneo e ordiniamo
            freq_df = pd.DataFrame({'word': words, 'count': counts})
            freq_df = freq_df[~freq_df['word'].isin(['https', 'com', 'www'])] # remove "https", "com", "www" if present
            top_10 = freq_df.sort_values('count', ascending=False).head(10)
            
            # Plot
            sns.barplot(data=top_10, x='count', y='word', ax=ax, palette='viridis', hue='word', legend=False)
            ax.set_title(f"Top 10 Tokens: {name}", fontsize=font_size+5, weight='bold')
            ax.set_xlabel("Frequency")
            ax.set_ylabel("")
            
        except ValueError:
            ax.text(0.5, 0.5, "No Data / Vocab Empty", ha='center', va='center')
            ax.set_title(f"{name} (Empty)")

    # Nascondiamo gli assi vuoti se ci sono (es. 5 dataset in griglia 2x3)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Most Frequent Tokens", fontsize=font_size+10, weight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


    # ---------------------------------------------------------
    # PLOT 6: Top Tokens Comparison (Divisi per Real e Fake)
    # ---------------------------------------------------------
    print("Generazione Top Tokens Charts (Split Real/Fake)...")
    
    # Configuriamo due passaggi: uno per Real (0) e uno per Fake (1)
    # (Label Value, Label Name, Color Palette Title)
    class_configs = [
        (0, 'Real', 'Greens_r'), 
        (1, 'Fake', 'Reds_r')
    ]

    for label_val, label_name, palette_name in class_configs:
        
        # Calcolo dimensioni griglia (identico a prima)
        n_datasets = len(dataset_dict)
        cols = 3 
        rows = math.ceil(n_datasets / cols)
        fig_h = rows * 4 
        
        fig, axes = plt.subplots(rows, cols, figsize=(18, fig_h))
        axes = axes.flatten()

        # Vectorizer (re-inizializzato per ogni classe per pulizia)
        count_vec = CountVectorizer(stop_words='english', max_features=5000, token_pattern=r"(?u)\b\w\w+\b")

        for i, (name, df) in enumerate(dataset_dict.items()):
            ax = axes[i]
            
            # FILTRO I DATI: Prendo solo le righe della classe corrente
            subset = df[df['labels'] == label_val]
            text_data = subset['texts'].fillna("")
            
            # Controllo se ci sono abbastanza dati (es. Kaggle_meg ha pochissimi Fake)
            if len(text_data) < 5:
                ax.text(0.5, 0.5, "Dati insufficienti", ha='center', va='center')
                ax.set_title(f"{name} ({label_name})")
                continue

            try:
                X = count_vec.fit_transform(text_data)
                counts = np.asarray(X.sum(axis=0)).flatten()
                words = count_vec.get_feature_names_out()
                
                # Creazione DF e Ordinamento
                freq_df = pd.DataFrame({'word': words, 'count': counts})
                freq_df = freq_df[~freq_df['word'].isin(['https', 'com', 'www'])] # remove "https", "com", "www" if present
                top_10 = freq_df.sort_values('count', ascending=False).head(10)
                
                # Plot
                sns.barplot(
                    data=top_10, 
                    x='count', 
                    y='word', 
                    ax=ax, 
                    palette=palette_name, 
                    hue='word', 
                    legend=False
                )
                
                ax.set_title(f"Top 10 Tokens ({label_name}): {name}", fontsize=font_size+5, weight='bold')
                ax.set_xlabel("Frequency")
                ax.set_ylabel("")
                
            except ValueError:
                # Gestione caso vocabolario vuoto o stop words totali
                ax.text(0.5, 0.5, "Empty Vocabulary", ha='center', va='center')
                ax.set_title(f"{name} ({label_name})")

        # Rimuovi assi vuoti
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # Titolo generale della figura (una per Real, una per Fake)
        plt.suptitle(f"Most Frequent Tokens - {label_name.upper()}", 
                     fontsize=font_size+10, weight='bold', y=1.02)
        plt.tight_layout()
        plt.show()




import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_thesis_plots_topic_date(dataset_dict, fig_size=(14, 8), font_size=15):
    """
    PLOT 1: Quantitative comparison between datasets divided by YEAR and Label (Real/Fake).
    """
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': font_size})
    
    all_data = []
    for name, df in dataset_dict.items():
        df_copy = df.copy()
        # Extract year: handles strings, NaT, or different formats
        df_copy['year'] = pd.to_datetime(df_copy['date'], errors='coerce').dt.year.fillna('N/A')
        # If the year is a number, convert it to int then string for the plot
        df_copy['year'] = df_copy['year'].apply(lambda x: str(int(x)) if isinstance(x, (int, float)) and not pd.isna(x) else str(x))
        
        df_copy['dataset_name'] = name
        all_data.append(df_copy)
    
    full_df = pd.concat(all_data, ignore_index=True)
    full_df['label_str'] = full_df['labels'].map({0: 'Real', 1: 'Fake'})

    # Utilizziamo catplot per creare facet (colonne) basate sulla Label
    g = sns.catplot(
        data=full_df, 
        kind="count",
        x="dataset_name", 
        hue="year", 
        col="label_str",
        palette="viridis",
        height=fig_size[1]/1.5, 
        aspect=fig_size[0]/fig_size[1],
        sharey=True
    )

    g.set_axis_labels("Topic", "Samples")
    g.set_titles("{col_name} News", fontsize=font_size+3)
    g.set_xticklabels(rotation=45)
    
    # Opzionale: scala logaritmica se i volumi sono molto diversi
    for ax in g.axes.flat:
        ax.set_yscale('log')
        
    plt.subplots_adjust(top=0.85)
    g.fig.suptitle("Distribution of Dates in each Topic and Class", fontsize=font_size+5)
    plt.show()

def generate_thesis_plots_date_topic(dataset_dict, fig_size=(14, 8), font_size=15):
    """
    PLOT 2: Confronto quantitativo tra dataset suddiviso per TOPIC e Label (Real/Fake).
    """
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': font_size})
    
    all_data = []
    for name, df in dataset_dict.items():
        df_copy = df.copy()
        df_copy['dataset_name'] = name
        all_data.append(df_copy)
    
    full_df = pd.concat(all_data, ignore_index=True)
    full_df['label_str'] = full_df['labels'].map({0: 'Real', 1: 'Fake'})

    # Utilizziamo catplot per creare facet (colonne) basate sulla Label
    g = sns.catplot(
        data=full_df, 
        kind="count",
        x="dataset_name", 
        hue="topic", 
        col="label_str",
        palette="Set2",
        height=fig_size[1]/1.5, 
        aspect=fig_size[0]/fig_size[1],
        sharey=True
    )

    g.set_axis_labels("Date", "Samples")
    g.set_titles("{col_name} News", fontsize=font_size+3)
    g.set_xticklabels(rotation=45)
    
    # Scala logaritmica per gestire dataset massivi (es. ISOT) e piccoli (es. Celebrity)
    for ax in g.axes.flat:
        ax.set_yscale('log')

    plt.subplots_adjust(top=0.85)
    g.fig.suptitle("Distribution of Topics in each Date and Class", fontsize=font_size+5)
    plt.show()