import pandas as pd
import numpy as np
from datasets import Dataset, concatenate_datasets

class ReplayBuffer:
    """
    Buffer that memorizes 'n' examples for each specific task.
    Internal structure: {task_name: (X_subset, y_subset)}
    """
    def __init__(self, samples_per_task=50, seed=42):
        self.samples_per_task = samples_per_task
        self.rng = np.random.default_rng(seed)
        self.buffer = {}  # Dict {task_name: (X, y)}
    
    def __len__(self):
        """Returns the total number of stored examples (sum of all tasks)."""
        return sum(len(v[1]) for v in self.buffer.values())
    
    def _get_balanced_indices(self, y, n_total):
        """
        Internal logic to calculate balanced indices.
        """
        # Convert y to numpy array for safe index calculations
        y_array = np.array(y)
        idx_0 = np.where(y_array == 0)[0]
        idx_1 = np.where(y_array == 1)[0]
        
        half_samples = n_total // 2
        
        # Base case: we have enough samples for both classes
        n_0 = min(len(idx_0), half_samples)
        n_1 = min(len(idx_1), half_samples)
        
        # If one class is scarce, compensate with the other to reach n_total
        if len(idx_0) < half_samples:
            n_1 = min(len(idx_1), n_total - len(idx_0))
        elif len(idx_1) < half_samples:
            n_0 = min(len(idx_0), n_total - len(idx_1))
            
        selected_0 = self.rng.choice(idx_0, size=n_0, replace=False)
        selected_1 = self.rng.choice(idx_1, size=n_1, replace=False)
        
        final_indices = np.concatenate([selected_0, selected_1])
        self.rng.shuffle(final_indices)
        return final_indices

    def add(self, task, X, y):
        """
        Adds data for a specific task.
        If the data exceeds 'samples_per_task', it selects a random subset.
        Overwrites if the task already exists (or you can choose to ignore).
        """
        n_samples = len(y)

        # If the data is small, keep it all
        if n_samples <= self.samples_per_task:
            self.buffer[task] = (X, y)
            return
        # else, calculate balanced indices
        indices = self._get_balanced_indices(y, self.samples_per_task)

        if isinstance(X, (pd.DataFrame, pd.Series)): # Pandas logic (for scikit-learn)
            # Ensure indices are reset to avoid slicing errors
            X = X.reset_index(drop=True)
            y = y.reset_index(drop=True)
  
            X_subset = X.iloc[indices].reset_index(drop=True)
            y_subset = y.iloc[indices].reset_index(drop=True)
            self.buffer[task] = (X_subset, y_subset)

        elif isinstance(X, Dataset): # HuggingFace Datasets logic (for Transformers)
            X_subset = X.select(indices)
            y_subset = y[indices]
            self.buffer[task] = (X_subset, y_subset)
        
        else: # Numpy logic (for TensorFlow)
            X_subset = X[indices]
            y_subset = y[indices]
            self.buffer[task] = (X_subset, y_subset)
        

    def get(self):
        """
        Returns ALL samples stored in the buffer (concatenated).
        Useful for replay training.
        """
        if not self.buffer:
            return None, None
        
        # Take the first element to determine the type
        first_X = list(self.buffer.values())[0][0]
        
        # Extract lists of X and y from dictionary values
        X_list = [v[0] for v in self.buffer.values()]
        y_list = [v[1] for v in self.buffer.values()]
        
        # Pandas concatenation
        if isinstance(first_X, (pd.DataFrame, pd.Series)):
            X_replay = pd.concat(X_list).reset_index(drop=True)
            y_replay = pd.concat(y_list).reset_index(drop=True)
            
        # Hugging Face Datasets concatenation
        elif isinstance(first_X, Dataset):
            X_replay = concatenate_datasets(X_list)
            y_replay = np.concatenate(y_list, axis=0)
            
        # Numpy concatenation
        else:
            X_replay = np.concatenate(X_list, axis=0)
            y_replay = np.concatenate(y_list, axis=0)
        
        return X_replay, y_replay