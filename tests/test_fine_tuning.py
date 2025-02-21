"""Test fine tuning pipeline."""

from datasets import load_dataset

from src.fine_tuning_pipeline import FineTunerPipeline, TaskType


def test_fine_tuning():
    """Test fine tuning pipeline."""
    dataset = load_dataset("xsum")
    _ = FineTunerPipeline(
        model_name="gpt2",
        dataset=dataset,
        mode=TaskType.TEXT_SUMMARISATION,
        fine_tuning_config={"text_column": "document", "target_column": "summary"},
    )
    breakpoint()
    return None


def test_peft():
    """Test PEFT pipeline."""
    return None
