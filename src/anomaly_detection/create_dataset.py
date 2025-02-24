"""Module to generate anomaly detection dataset."""

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def generate_anomaly_dataset(
    n_samples: int = 5000,
    n_features: int = 10,
    n_categorical: int = 3,
    anomaly_ratio: float = 0.02,
    random_state: int = 42,
    sequence_length: int = 5,
    output_size: int = 2,
) -> tuple([torch.Tensor, torch.Tensor]):
    """Generate anomaly detection dataset.

    Function to generate dataset that can be used for anomaly detection

    :param n_samples: Total number of samples.
    :param n_features: Number of numerical features.
    :param n_categorical: Number of categorical features.
    :param anomaly_ratio: Fraction of anomalies (minority class).
    :param random_state: Random seed for reproducibility.
    :param sequence_length: Number of time steps per sequence.
    :param output_size: Number of classes (for classification).
    :return: (x, y) PyTorch tensors formatted for LSTM.

    """
    np.random.seed(random_state)

    adjusted_n_samples = (n_samples // sequence_length) * sequence_length

    # Generate numerical data
    x, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_clusters_per_class=1,
        weights=[1 - anomaly_ratio, anomaly_ratio],
        random_state=random_state,
    )

    # Reshape for LSTM format: (batch_size, sequence_length, input_size)
    batch_size = adjusted_n_samples // sequence_length
    x = x.reshape(batch_size, sequence_length, n_features)

    y = y.reshape(batch_size, sequence_length)

    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    return x_tensor, y_tensor


# TODO: Refactor this
def preprocess_api_dataset(dataset_name: str, target_name: str, sequence_length: int = 5):
    """Preprocess the API dataset."""
    df = pd.read_csv(dataset_name)

    df = df.fillna(0)

    xtrain, xtest, ytrain, ytest = train_test_split(
        df.drop(columns=target_name), df[target_name], test_size=0.33, random_state=42
    )

    xtrain = reshape_data(xtrain, sequence_length=sequence_length)
    xtest = reshape_data(xtest, sequence_length=sequence_length)

    le = LabelEncoder()
    ytrain = le.fit_transform(ytrain)
    ytrain = reshape_data(pd.DataFrame(ytrain), sequence_length=sequence_length, if_target=True)

    ytest = le.fit_transform(ytest)
    ytest = reshape_data(pd.DataFrame(ytest), sequence_length=sequence_length, if_target=True)

    return xtrain, ytrain, xtest, ytest


# TODO: Refactor this
def reshape_data(df, sequence_length=10, batch_size=None, if_target=False):
    """Reshape the data.

    Automatically calculates the correct shape (batch_size, sequence_length, features)
    for an LSTM and reshapes the DataFrame into a PyTorch tensor.

    :param df: Pandas DataFrame with numerical features
    :param sequence_length: Number of timesteps per sequence
    :param batch_size: Optional, if None it will be auto-calculated
    :return: LSTM-ready PyTorch tensor (batch_size, sequence_length, features)
    """
    # Drop non-numeric columns
    df = df.select_dtypes(include=[np.number])

    n_samples, n_features = df.shape

    # Auto-calculate batch size if not provided
    if batch_size is None:
        batch_size = n_samples // sequence_length  # Ensure full sequences
        n_samples = batch_size * sequence_length  # Trim excess

    if n_samples % sequence_length != 0:
        print(f"⚠️ Warning: Trimming {n_samples % sequence_length} samples to fit sequence length.")
        n_samples = (n_samples // sequence_length) * sequence_length

    # Reshape data to (batch_size, sequence_length, features)
    if if_target:
        reshaped_data = df.values[:n_samples].reshape(batch_size, sequence_length)
        return torch.tensor(reshaped_data, dtype=torch.long)
    else:
        reshaped_data = df.values[:n_samples].reshape(batch_size, sequence_length, n_features)
        return torch.tensor(reshaped_data, dtype=torch.float32)
