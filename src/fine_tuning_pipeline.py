"""Fine-tuning pipeline for transformers models."""

from enum import Enum

from datasets import DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


class ModelType(Enum):
    """Enum class for model types."""

    sequence_classification = AutoModelForSequenceClassification
    text_generation = AutoModelForCausalLM


class FineTunerPipeline:
    """Fine-tuning pipeline for transformers models."""

    def __init__(self, model_name: str, dataset: DatasetDict, mode: ModelType):
        """Initialize the fine-tuning pipeline."""
        self.model_name = model_name
        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = mode.from_pretrained(self.model_name)
