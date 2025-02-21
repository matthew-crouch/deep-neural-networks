"""Test fine tuning pipeline."""

import pytest
from datasets import load_dataset
from evaluate import load

from src.fine_tuning_pipeline import FineTunerPipeline, TaskType


def test_tokenisation():
    """Test tokenisation function in the fine_tuner pipeline."""
    dataset = load_dataset("xsum")
    _ = load("rouge")

    fine_tuner = FineTunerPipeline(
        dataset=dataset,
        mode=TaskType.TEXT_SUMMARISATION,
        fine_tuning_config={"text_column": "document", "target_column": "summary"},
    )
    fine_tuner.tokenize()


@pytest.mark.skip(reason="Not implemented")
def test_peft():
    """Test PEFT pipeline."""
    return None
