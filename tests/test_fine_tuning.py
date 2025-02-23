"""Test fine tuning pipeline."""

import pytest
import torch
import transformers
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


@pytest.mark.skip("Test for documentation")
@pytest.mark.parametrize("training_mode", ["cpu", "gpu", "multi-gpu"])
def test_peft(training_mode):
    """Test PEFT pipeline."""
    dataset = load_dataset("xsum", trust_remote_code=True)
    _ = load("rouge")

    ft_pipeline = FineTunerPipeline(
        mode=TaskType.TEXT_SUMMARISATION,
        fine_tuning_config={"text_column": "document", "target_column": "summary"},
    )
    breakpoint()
    _ = ft_pipeline.run(dataset=dataset)


@pytest.mark.skip("Test for documentation")
def test_pipeline():
    """Test huggingface pipeline."""
    model_id = "meta-llama/Meta-Llama-3-8B"
    _ = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
