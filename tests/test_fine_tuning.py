"""Test fine tuning pipeline."""

import pytest
from datasets import load_dataset
from evaluate import load

from src.fine_tuning_pipeline import FineTunerPipeline, TaskType


def test_tokenisation():
    """Test tokenisation function in the fine_tuner pipeline."""
    dataset = load_dataset("xsum", trust_remote_code=True)
    _ = load("rouge")

    fine_tuner = FineTunerPipeline(
        mode=TaskType.TEXT_SUMMARISATION,
        fine_tuning_config={"text_column": "document", "target_column": "summary"},
    )
    fine_tuner.dataset = dataset
    fine_tuner.tokenize()


@pytest.mark.skip(reason="Under development")
def test_peft():
    """Test PEFT pipeline."""
    dataset = load_dataset("xsum", trust_remote_code=True)
    _ = load("rouge")

    ft_pipeline = FineTunerPipeline(
        mode=TaskType.TEXT_SUMMARISATION,
        fine_tuning_config={"text_column": "document", "target_column": "summary"},
    )
    _ = ft_pipeline.run(dataset=dataset)
