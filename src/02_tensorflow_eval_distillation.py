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
from utils.utils_continual.utils_distillation import *

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import json

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from gensim.models import KeyedVectors

# Ignore warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

TASK_TYPE = "date" # specify the task type ("topic" or "date")
MODEL = "BiLSTM" # specify the model type ("CNN" or "BiLSTM")
TEMPERATURE = 2.0
ALPHA = 0.5
RESULTS_DIR_PATH = f"results_distillation/results_{MODEL}_{TASK_TYPE}"
os.makedirs(RESULTS_DIR_PATH, exist_ok=True)



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
    for task, df in datasets.items():
        seq = tokenizer.texts_to_sequences(df["texts"].astype(str).tolist()) # convert texts to sequences of integers
        X = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post') # padd/truncate sequences to max_len
        y = encoder.transform(df["labels"].values) # encode labels to integers

        # split dataset in train/val/test
        processed_datasets[task] = split_dataset_tensorflow(X, y)
        
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

def build_model(model, **kwargs):
    """
    Build and compile the specified model.

    Args:
        model (str): Type of model to build ("CNN" or "BiLSTM")
        **kwargs: Additional arguments for the model building functions
    
    Returns:
        model: Compiled Keras model
    """
    if model == "CNN":
        return build_model_CNN(**kwargs)
    elif model == "BiLSTM":
        return build_model_BiLSTM(**kwargs)
    else:
        raise ValueError("Invalid model type. Choose either 'CNN' or 'BiLSTM'.")

def build_model_CNN(embedding_matrix, num_words, embedding_dim=300,
                filter_size=4, num_filters=96, dropout=0.4,
                hidden_units=32, learning_rate=1e-4):
    """
    Build and compile the CNN model.

    Args:
        embedding_matrix: Pre-trained embedding matrix
        num_words: Number of words in the vocabulary
        embedding_dim: Dimension of word embeddings
        filter_size: Size of the convolutional filters
        num_filters: Number of convolutional filters
        dropout: Dropout rate
        hidden_units: Number of units in the dense hidden layer
        learning_rate: Learning rate for the Adam optimizer

    Returns:
        model: Compiled Keras CNN model
    """
    model = Sequential([
        Embedding(num_words,
                  embedding_dim,
                  weights=[embedding_matrix],
                  trainable=False # freeze embeddings
        ),
        Dropout(dropout),
        Conv1D(num_filters, filter_size, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(hidden_units, activation='relu'),
        Dropout(dropout),
        Dense(1)
    ])

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def build_model_BiLSTM(embedding_matrix, num_words, embedding_dim=300,
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
        Dense(1)
    ])

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model



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



# ------------------------------------
# Fine-tuning on Dataset by Topic/Date
# ------------------------------------

path = f"{RESULTS_DIR_PATH}/best_params"
with open(f"{path}.json", "r") as f:
    best_params = json.load(f)

current_model = build_model(
    model=MODEL,
    embedding_matrix=embedding_matrix,
    num_words=num_words,
    embedding_dim=300,
    **best_params
)

results_task = {}
results_cumulative = {}
results_full = {}

# test set for balanced cumulative/full evaluation (via sample weights)
X_cumulative_test = []
y_cumulative_test = []
X_full_test = np.concatenate([datasets[t]["test"][0] for t in datasets])
y_full_test = np.concatenate([datasets[t]["test"][1] for t in datasets])
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

teacher_model = None

# sequential training
for i, (task, data) in enumerate(datasets.items()):
    print(f"\n=== Phase {i+1}: Training/Fine-tuning on {TASK_TYPE}: {task} ===")

    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    X_test, y_test = data["test"]

    # Distiller for Knowledge Distillation
    distiller = Distiller(
        student=current_model, 
        teacher=teacher_model, 
        alpha=ALPHA, 
        temperature=TEMPERATURE
    )
    
    distiller.compile(
        optimizer=Adam(learning_rate=best_params["learning_rate"]),
        metrics=['accuracy']
    )

    # early stopping
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

    # fine-tune on train + val
    distiller.fit(
        X_train, y_train,
        epochs=50,
        batch_size=124,
        validation_data=(X_val, y_val),
        callbacks=[es],
        verbose=1
    )

    # Teacher for the next task: cloning current model and freezing weights
    teacher_model = tf.keras.models.clone_model(current_model)
    teacher_model.set_weights(current_model.get_weights())
    teacher_model.trainable = False

    y_pred = current_model.predict(X_test)
    y_pred = tf.nn.sigmoid(y_pred).numpy() # apply sigmoid to logits
    y_pred = (y_pred > 0.5).astype(int)
    print(f"Classification Report after {TASK_TYPE} {task}:")
    print(classification_report(y_test, y_pred))
    print(f"Confusion Matrix after {TASK_TYPE} {task}:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\nWeighted F1-score after {TASK_TYPE} {task}:", f1_score(y_test, y_pred, average="weighted"))

    # evaluation on all topics/dates
    print(f"\n--- Evaluation on all {TASK_TYPE}s ---")
    results_task[task] = {}
    for test_task, test_data in datasets.items(): # for each topic/date
        X_te, y_te = test_data["test"]
        preds = current_model.predict(X_te)
        preds = tf.nn.sigmoid(preds).numpy() # apply sigmoid to logits
        preds = (preds > 0.5).astype(int)
        f1 = f1_score(y_te, preds, average="weighted")
        results_task[task][test_task] = f1
        print(f"Evaluation on {TASK_TYPE} {test_task}: Weighted F1 = {f1:.4f}")
    
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
    X_cum_test_concat = np.concatenate(X_cumulative_test, axis=0)
    y_cum_test_concat = np.concatenate(y_cumulative_test, axis=0)
    sample_weights_cum_concat = np.concatenate(sample_weights_cumulative)
    # predict and evaluate
    preds_cumulative = current_model.predict(X_cum_test_concat)
    preds_cumulative = tf.nn.sigmoid(preds_cumulative).numpy() # apply sigmoid to logits
    preds_cumulative = (preds_cumulative > 0.5).astype(int)
    results_cumulative[task] = f1_score(y_cum_test_concat, preds_cumulative, average="weighted", sample_weight=sample_weights_cum_concat)
    print(f"Evaluation on cumulative test set after {TASK_TYPE} {task}: Weighted F1 = {results_cumulative[task]:.4f}")
    
    # evaluation on full test set
    print("\n--- Evaluation on full test set ---")
    preds_full = current_model.predict(X_full_test)
    preds_full = tf.nn.sigmoid(preds_full).numpy() # apply sigmoid to logits
    preds_full = (preds_full > 0.5).astype(int)
    f1_full = f1_score(y_full_test, preds_full, average="weighted", sample_weight=sample_weights_full)
    results_full[task] = f1_full
    print(f"Evaluation on full test set after {TASK_TYPE} {task}: Weighted F1 = {f1_full:.4f}")



# ---------------
# Results summary
# ---------------
print_results_summary(results_task, results_full, type=TASK_TYPE.capitalize())

# save results for Single Topic/Date
path = f"{RESULTS_DIR_PATH}/results_single_{TASK_TYPE}_{str(ALPHA).replace('.', '')}.json"
with open(path, "w") as f:
    json.dump(results_task, f, indent=4)

# save results for Cumulative Test Set
path = f"{RESULTS_DIR_PATH}/results_cumulative_{TASK_TYPE}_{str(ALPHA).replace('.', '')}.json"
with open(path, "w") as f:
    json.dump(results_cumulative, f, indent=4)

# save results for Full Test Set
path = f"{RESULTS_DIR_PATH}/results_full_{TASK_TYPE}_{str(ALPHA).replace('.', '')}.json"
with open(path, "w") as f:
    json.dump(results_full, f, indent=4)



# -----------------------------------------
# Plot on each Topic/Date and Full Test Set
# -----------------------------------------

path = f"{RESULTS_DIR_PATH}/f1_by_{TASK_TYPE}_{str(ALPHA).replace('.', '')}.png"
plot_f1_by_task(results_task, type=TASK_TYPE.capitalize(), path=path)

path = f"{RESULTS_DIR_PATH}/f1_cumulative_test_{TASK_TYPE}_{str(ALPHA).replace('.', '')}.png"
plot_f1_cumulative_test(results_cumulative, type=TASK_TYPE.capitalize(), path=path)

path = f"{RESULTS_DIR_PATH}/f1_full_test_{TASK_TYPE}_{str(ALPHA).replace('.', '')}.png"
plot_f1_full_test(results_full, type=TASK_TYPE.capitalize(), path=path)