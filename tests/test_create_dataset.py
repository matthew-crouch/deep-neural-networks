"""Tests for the create_dataset module."""

from src.create_dataset import generate_anomaly_dataset


def test_generate_anomaly_dataset():
    """Test the generate_anomaly_dataset function."""
    df = generate_anomaly_dataset(n_samples=15000)
    assert not df.empty
