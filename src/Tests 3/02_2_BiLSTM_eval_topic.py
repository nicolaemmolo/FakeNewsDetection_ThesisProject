import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # "0" o "1"

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("Using GPU:", gpus[0])
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found, using CPU.")

from utils.utils_data_preprocessing import *
from utils.utils import *

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import json

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from gensim.models import KeyedVectors

# Ignore warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

RESULTS_DIR_PATH = "results/results_BiLSTM_topic"
bool_150 = False
bool_150test = False
bool_balanced_test = False
bool_inverted = False
PATH_SUFFIX = create_path_suffix(
    bool_150=bool_150,
    bool_150test=bool_150test,
    bool_balanced_test=bool_balanced_test,
    bool_inverted=bool_inverted
)



# --------------------------
# Data Tokenization function
# --------------------------

def prepare_data(datasets, max_words=20000, max_len=300):
    """
    Tokenization and label encoding for multiple datasets.

    Args:
        datasets (dict): A dictionary where keys are dataset names and values are pandas DataFrames with 'texts' and 'labels' columns.
        max_words (int): Maximum number of words to keep in the tokenizer vocabulary.
        max_len (int): Maximum length of sequences after padding/truncating.

    Returns:
        processed_datasets (dict): A dictionary with the same keys as input, where each value is another dict with 'train', 'val', 'test' splits containing (X, y) tuples.
        tokenizer (Tokenizer): Fitted Keras Tokenizer.
        encoder (LabelEncoder): Fitted sklearn LabelEncoder.
    """
    # Unisci tutti i testi di tutti i dataset
    all_texts = []
    for df in datasets.values():
        all_texts.extend(df["texts"].astype(str).tolist())

    # Tokenizer
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>") # tokenizer with OOV token
    tokenizer.fit_on_texts(all_texts) # create vocabulary {word: index}

    # LabelEncoder globale
    all_labels = np.concatenate([df["labels"].values for df in datasets.values()])
    encoder = LabelEncoder().fit(all_labels)

    # Applica tokenizzazione e encoding a ogni dataset
    processed_datasets = {}
    for topic, df in datasets.items():
        seq = tokenizer.texts_to_sequences(df["texts"].astype(str).tolist()) # convert texts to sequences of integers
        X = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post') # padd/truncate sequences to max_len
        y = encoder.transform(df["labels"].values) # encode labels to integers

        if bool_150: # split dataset in train/val/test (max 150 samples in val and test)
            processed_datasets[topic] = split_dataset_tensorflow_150(X, y)
        else: # split dataset in train/val/test
            processed_datasets[topic] = split_dataset_tensorflow(X, y)

    return processed_datasets, tokenizer, encoder



# ---------------------------------
# Load Word2Vec embeddings function
# ---------------------------------

def load_word2vec(tokenizer, max_words=20000, embedding_dim=300):
    """
    Load pre-trained Word2Vec embeddings and create embedding matrix

    Args:
        tokenizer: Keras Tokenizer object with fitted vocabulary
        max_words: maximum number of words to consider from tokenizer
        embedding_dim: dimension of Word2Vec embeddings

    Returns:
        embedding_matrix: numpy array of shape (num_words, embedding_dim)
        num_words: actual number of words considered
    """
    
    print("\nLoading pre-trained Word2Vec model (may take time)...")
    w2v_path = "../Word2Vec_GoogleNews300/word2vec-google-news-300.model"
    w2v_model = KeyedVectors.load(w2v_path, mmap='r') # load model with memory mapping (mmap='r' for only reading) 

    word_index = tokenizer.word_index               # vocabulary from tokenizer {word: index}
    num_words = min(max_words, len(word_index) + 1) # number of words to consider (max_words or vocab size)

    # create embedding matrix: each row corresponds to a word index from tokenizer, each column to an embedding dimension
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        if i >= max_words: # skip because we only consider top max_words
            continue
        if word in w2v_model: # if word has a pre-trained embedding
            embedding_matrix[i] = w2v_model[word]

    return embedding_matrix, num_words



# -----------------------
# Model building function
# -----------------------

def build_model(embedding_matrix, num_words, embedding_dim=300,
                num_units=96, dropout=0.4,
                hidden_units=32, learning_rate=1e-4):
    """
    Build and compile the BiLSTM model.

    Args:
        embedding_matrix: Pre-trained embedding matrix
        num_words: Number of words in the vocabulary
        embedding_dim: Dimension of word embeddings
        num_units: Number of LSTM units
        dropout: Dropout rate
        hidden_units: Number of units in the dense hidden layer
        learning_rate: Learning rate for the Adam optimizer

    Returns:
        model: Compiled Keras BiLSTM model
    """

    model = Sequential([
        Embedding(
            num_words,
            embedding_dim,
            weights=[embedding_matrix],
            trainable=True),
        Bidirectional(LSTM(num_units, return_sequences=False, dropout=dropout, recurrent_dropout=0.0)),
        Dense(hidden_units, activation='relu'),
        Dropout(dropout),
        Dense(1, activation='sigmoid')
    ])

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model



# ----------------
# Data Preparation
# ----------------

dataset_df = data_by_topic(inverted=bool_inverted)

print("--- DATASET SIZES BY TOPIC ---")
for topic, df in dataset_df.items():
    print(f"Topic: {topic}, Number of samples: {len(df)}")

datasets, tokenizer, encoder = prepare_data(dataset_df)
embedding_matrix, num_words = load_word2vec(tokenizer)

print("--- DATA SPLITS SIZES BY TOPIC ---")
for topic, data_splits in datasets.items():
    print(f"Topic: {topic}")
    print(f"Train: {len(data_splits['train'][0])}, Val: {len(data_splits['val'][0])}, Test: {len(data_splits['test'][0])}")



# -------------------------------
# Fine-tuning on Dataset by Topic
# -------------------------------

path = f"{RESULTS_DIR_PATH}/best_params"
if bool_150: path += "_150"
with open(f"{path}.json", "r") as f:
    best_params = json.load(f)

best_model = build_model(
    embedding_matrix,
    num_words,
    embedding_dim=300,
    num_units=best_params["num_units"],
    dropout=best_params["dropout"],
    hidden_units=best_params["hidden_units"],
    learning_rate=best_params["learning_rate"]
)

results_topic = {}
results_full = {}

if bool_150test: # balanced full test set (max length 150)
    X_trimmed_list = []
    y_trimmed_list = []
    for t in datasets:
        X_test, y_test = datasets[t]["test"]
        X_trim, y_trim, _, _ = trim_and_extract(X_test, y_test, max_len=150)
        X_trimmed_list.append(X_trim)
        y_trimmed_list.append(y_trim)
    X_full_test = np.concatenate(X_trimmed_list)
    y_full_test = np.concatenate(y_trimmed_list)
elif bool_balanced_test: # balanced full test set (via sample weights)
    X_full_test = np.concatenate([datasets[t]["test"][0] for t in datasets])
    y_full_test = np.concatenate([datasets[t]["test"][1] for t in datasets])
    sample_weights = []
    for t in datasets:
        _, y_test_t = datasets[t]["test"]
        n = len(y_test_t)
        w = 1.0 / n  # weight for each sample in this topic
        sample_weights.extend([w] * n)
    sample_weights = np.array(sample_weights)
else: # unbalanced full test set
    X_full_test = np.concatenate([datasets[t]["test"][0] for t in datasets])
    y_full_test = np.concatenate([datasets[t]["test"][1] for t in datasets])

# sequential training
for i, (topic, data) in enumerate(datasets.items()):
    print(f"\n=== Phase {i+1}: Training/Fine-tuning on topic: {topic} ===")

    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    X_test, y_test = data["test"]

    # early stopping
    es = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True, verbose=1)

    # fine-tune on train + val
    best_model.fit(
        np.concatenate([X_train, X_val]),
        np.concatenate([y_train, y_val]),
        epochs=50,
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=[es],
        verbose=1
    )

    y_pred = best_model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)
    print(f"Classification Report after topic {topic}:")
    print(classification_report(y_test, y_pred))
    print(f"Confusion Matrix after topic {topic}:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\nWeighted F1-score after topic {topic}:", f1_score(y_test, y_pred, average="weighted"))

    # evaluation on all topics
    print("\n--- Evaluation on all topics ---")
    results_topic[topic] = {}
    for test_topic, test_data in datasets.items(): # for each topic
        X_te, y_te = test_data["test"]
        preds = best_model.predict(X_te)
        preds = (preds > 0.5).astype(int)
        f1 = f1_score(y_te, preds, average="weighted")
        results_topic[topic][test_topic] = f1
        print(f"Evaluation on topic {test_topic}: Weighted F1 = {f1:.4f}")
    
    # evaluation on full test set
    print("\n--- Evaluation on full test set ---")
    preds_full = best_model.predict(X_full_test)
    preds_full = (preds_full > 0.5).astype(int)
    if bool_balanced_test:
        f1_full = f1_score(y_full_test, preds_full, average="weighted", sample_weight=sample_weights)
    else:
        f1_full = f1_score(y_full_test, preds_full, average="weighted")
    results_full[topic] = f1_full
    print(f"Evaluation on full test set after topic {topic}: Weighted F1 = {f1_full:.4f}")



# ---------------
# Results summary
# ---------------

print_results_summary(results_topic, results_full, type="Topic")

# save results for Single Topic
path = f"{RESULTS_DIR_PATH}/results_single_topic{PATH_SUFFIX}.json"
with open(path, "w") as f:
    json.dump(results_topic, f, indent=4)

# save results for Full Test Set
path = f"{RESULTS_DIR_PATH}/results_full_topic{PATH_SUFFIX}.json"
with open(path, "w") as f:
    json.dump(results_full, f, indent=4)



# ------------------------------------
# Plot on each Topic and Full Test Set
# ------------------------------------

path = f"{RESULTS_DIR_PATH}/f1_by_topic{PATH_SUFFIX}.png"
plot_f1_by_task(results_topic, type="Topic", path=path)

path = f"{RESULTS_DIR_PATH}/f1_full_test_topic{PATH_SUFFIX}.png"
plot_f1_full_test(results_full, type="Topic", path=path)