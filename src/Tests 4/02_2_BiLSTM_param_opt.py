import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # "0" o "1"

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
from sklearn.metrics import f1_score
import json

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from gensim.models import KeyedVectors

import optuna
from functools import partial
from sklearn.utils import shuffle

# Ignore warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

TASK_TYPE = "date" # specify the task type ("topic" or "date")
RESULTS_DIR_PATH = f"results/results_BiLSTM_{TASK_TYPE}"
bool_150 = False



# --------------------------
# Data Tokenization function
# --------------------------

def prepare_data(datasets, max_words=30000, max_len=300):
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
    # Merge all texts from all datasets
    all_texts = []
    for df in datasets.values():
        all_texts.extend(df["texts"].astype(str).tolist())

    # Tokenizer
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>") # tokenizer with OOV token
    tokenizer.fit_on_texts(all_texts) # create vocabulary {word: index}

    # Global LabelEncoder
    all_labels = np.concatenate([df["labels"].values for df in datasets.values()])
    encoder = LabelEncoder().fit(all_labels)

    # Apply tokenization and encoding to each dataset
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

def load_word2vec(tokenizer, max_words=30000, embedding_dim=300):
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
            trainable=False # freeze embeddings
        ),
        Bidirectional(LSTM(num_units, return_sequences=False, dropout=dropout, recurrent_dropout=0.0)),
        Dense(hidden_units, activation='relu'),
        Dropout(dropout),
        Dense(1, activation='sigmoid')
    ])

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model



# -------------------------
# Optuna objective function
# -------------------------

def objectiveBiLSTM(trial, X_train, y_train, X_val, y_val, embedding_matrix, num_words, embedding_dim=300):
    """
    Optuna objective function for hyperparameter optimization of BiLSTM model.

    Args:
        trial: Optuna trial object
        X_train: Training data features
        y_train: Training data labels
        X_val: Validation data features
        y_val: Validation data labels
        embedding_matrix: Pre-trained embedding matrix
        num_words: Number of words in the vocabulary
        embedding_dim: Dimension of word embeddings

    Returns:
        f1: Weighted F1-score on validation data
    """

    num_units = trial.suggest_categorical("num_units", [16, 32, 64, 96, 128])            # number of LSTM units
    dropout = trial.suggest_categorical("dropout", [0.0, 0.2, 0.4, 0.6, 0.8])            # dropout rate
    hidden_units = trial.suggest_categorical("hidden_units", [8, 16, 32, 64])            # number of units in dense layer
    learning_rate = trial.suggest_categorical("learning_rate", [1e-5, 1e-4])             # (no 1e-3, 1e-2) learning rate for Adam optimizer

    model = build_model(
        embedding_matrix=embedding_matrix,
        num_words=num_words,
        embedding_dim=embedding_dim,
        num_units=num_units,
        dropout=dropout,
        hidden_units=hidden_units,
        learning_rate=learning_rate
    )

    # train model with early stopping
    es = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True, verbose=0)
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=512,
        callbacks=[es], # early stopping
        verbose=0
    )
    preds = (model.predict(X_val) > 0.5).astype(int)
    f1 = f1_score(y_val, preds)
    return f1



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

datasets, tokenizer, encoder = prepare_data(dataset_df)
embedding_matrix, num_words = load_word2vec(tokenizer)

print(f"--- DATA SPLITS SIZES BY {TASK_TYPE.upper()} ---")
for task, data_splits in datasets.items():
    print(f"{TASK_TYPE.capitalize()}: {task}")
    print(f"Train: {len(data_splits['train'][0])}, Val: {len(data_splits['val'][0])}, Test: {len(data_splits['test'][0])}")



# ---------------------------
# Hyperparameter optimization
# ---------------------------

# merge all training and validation data across topics/dates for optimization
X_train = np.concatenate([datasets[task]['train'][0] for task in datasets])
y_train = np.concatenate([datasets[task]['train'][1] for task in datasets])
X_val = np.concatenate([datasets[task]['val'][0] for task in datasets])
y_val = np.concatenate([datasets[task]['val'][1] for task in datasets])
# shuffle the combined training and validation data
X_train, y_train = shuffle(X_train, y_train, random_state=42)
X_val, y_val = shuffle(X_val, y_val, random_state=42)

# functools.partial to pass data to objective function
objective_with_data = partial(objectiveBiLSTM,
                              X_train=X_train,
                              X_val=X_val,
                              y_train=y_train,
                              y_val=y_val,
                              embedding_matrix=embedding_matrix,
                              num_words=num_words)

study = optuna.create_study(direction="maximize") # maximize F1-score
study.optimize(objective_with_data, n_trials=50) # 50 trials for demonstration; increase for better results

print("Best parameters:", study.best_params)

path = f"{RESULTS_DIR_PATH}/best_params"
if bool_150: path += "_150"
with open(f"{path}.json", "w") as f:
    json.dump(study.best_params, f, indent=4)