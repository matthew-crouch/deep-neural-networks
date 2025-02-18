"""Tests for the create_dataset module."""

from src.create_dataset import generate_anomaly_dataset


def test_generate_anomaly_dataset():
    """Test the generate_anomaly_dataset function."""
    x, y = generate_anomaly_dataset(n_samples=1500)
    assert x.shape[0] == 300
    assert x.shape[1] == 5
    assert x.shape[2] == 10

    assert y.shape[0] == 300
    assert y.shape[1] == 5
