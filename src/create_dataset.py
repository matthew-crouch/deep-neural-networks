"""Module to generate anomaly detection dataset."""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def generate_anomaly_dataset(
    n_samples: int = 5000,
    n_features: int = 10,
    n_categorical: int = 3,
    anomaly_ratio: float = 0.02,
    random_state: int = 42,
):
    """Generate anomaly detection dataset.

    Function to generate dataset that can be used for anomaly detection

    :param n_samples: Total number of samples in the dataset.
    :param n_features: Number of numerical features.
    :param n_categorical: Number of categorical features.
    :param anomaly_ratio: Fraction of anomalies (minority class).
    :param random_state: Random seed for reproducibility.
    :return: A pandas DataFrame with numerical and categorical features, and a 'target' column.

    """
    np.random.seed(random_state)

    n_anomalies = int(n_samples * anomaly_ratio)
    n_normal = n_samples - n_anomalies

    # Generate numerical data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_clusters_per_class=1,
        weights=[1 - anomaly_ratio, anomaly_ratio],
        random_state=random_state,
    )

    # Convert numerical data to DataFrame
    df = pd.DataFrame(X, columns=[f"num_feature_{i}" for i in range(n_features)])

    for i in range(n_categorical):
        normal_categories = np.random.choice(["A", "B", "C"], size=n_normal, p=[0.5, 0.3, 0.2])
        anomaly_categories = np.random.choice(["D", "E"], size=n_anomalies)
        df[f"cat_feature_{i}"] = np.concatenate([normal_categories, anomaly_categories])

    # Add target column
    df["target"] = y

    return df
