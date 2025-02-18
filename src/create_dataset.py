"""Module to generate anomaly detection dataset."""

import numpy as np
import torch
from sklearn.datasets import make_classification


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
