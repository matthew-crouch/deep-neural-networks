"""Test fine tuning pipeline."""

import pytest
import torch
import transformers
from datasets import load_dataset
from evaluate import load
from peft import LoraConfig

from src.fine_tuning_pipeline import FineTunerPipeline, FineTuningConfig, TaskType
from src.pipelines.tokenizer import Tokenizer


def test_tokenisation():
    """Test tokenisation function in the fine_tuner pipeline."""
    dataset = load_dataset("xsum", trust_remote_code=True)

    fine_tuning_config = {
        "ft_model_name": "custom_model",
        "text_column": "document",
        "target_column": "summary",
    }
    tokenizer = Tokenizer("google/pegasus-xsum", config=FineTuningConfig(**fine_tuning_config))
    train_data, eval_data = tokenizer.tokenize(dataset=dataset)


@pytest.mark.skip("Test for documentation")
@pytest.mark.parametrize("training_mode", ["cpu"])
def test_peft(training_mode):
    """Test PEFT pipeline."""
    dataset = load_dataset("xsum", trust_remote_code=True)
    _ = load("rouge")

    ft_pipeline = FineTunerPipeline(
        mode=TaskType.TEXT_SUMMARISATION,
        fine_tuning_config={
            "ft_model_name": "custom_model",
            "text_column": "document",
            "target_column": "summary",
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 2,
            "sample_size": 10,
            "lora": {
                "enabled": True,
                "lora_config": LoraConfig(
                    r=8,
                    lora_alpha=32,
                    lora_dropout=0.1,
                    bias="none",
                    target_modules=["q_proj", "v_proj"],
                ),
            },
        },
    )
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
