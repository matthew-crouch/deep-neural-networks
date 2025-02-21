"""Test fine tuning pipeline."""

from datasets import load_dataset
from evaluate import load

from src.fine_tuning_pipeline import FineTunerPipeline, TaskType


def test_fine_tuning():
    """Test fine tuning pipeline."""
    dataset = load_dataset("xsum")
    eval_dataset = load("rouge")

    fine_tuner = FineTunerPipeline(
        dataset=dataset,
        mode=TaskType.TEXT_SUMMARISATION,
        fine_tuning_config={"text_column": "document", "target_column": "summary"},
    )
    breakpoint()
    return None


def test_peft():
    """Test PEFT pipeline."""
    return None
