"""Fine-tuning pipeline for transformers models."""

from enum import Enum

from datasets import DatasetDict
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Seq2SeqTrainer,
)


class TaskType(Enum):
    """Enum class for model types."""

    SEQUENCE_CLASSIFICATION = "sequence_classification"
    TEXT_GENERATION = "text_generation"
    TEXT_SUMMARISATION = "summarisation"


class FineTuningConfig(BaseModel):
    """Dataclass for fine-tuning configuration."""

    text_column: str
    target_column: str
    lora: bool = False
    quantisation: bool = False


class FineTunerPipeline:
    """Fine-tuning pipeline for transformers models.

    This pipeline supports and simplifies fine tuning transformer models
    for specific tasks. The pipeline supports sequence classification,
    text generation, and summarisation tasks.
    """

    def __init__(
        self,
        dataset: DatasetDict,
        mode: TaskType,
        fine_tuning_config: dict,
    ):
        """Initialize the fine-tuning pipeline."""
        self.dataset = dataset

        transformers = self._mode_options.get(mode)
        transformer_model, model_name, model_kwargs = (
            transformers.get("task"),
            transformers.get("models"),
            transformers.get("model_kwargs"),
        )

        self.fine_tuning_config = FineTuningConfig(**fine_tuning_config)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = transformer_model.from_pretrained(
            model_name, torch_dtype="auto", **model_kwargs
        )

    # TODO: Eventually we could look to abstract this out to a base class
    @property
    def _mode_options(self):
        """Model options for the pipeline."""
        return {
            TaskType.SEQUENCE_CLASSIFICATION: {
                "task": AutoModelForSequenceClassification,
                "models": "bert",
                "model_kwargs": {"num_labels": 2},
            },
            TaskType.TEXT_GENERATION: AutoModelForCausalLM,
            TaskType.TEXT_SUMMARISATION: {
                "task": AutoModelForSeq2SeqLM,
                "models": "google/pegasus-xsum",
                "model_kwargs": {},
                "trainer": {
                    "type": Seq2SeqTrainer,
                    "trainer_kwargs": {"evaluation_strategy": "epoch"},
                },
            },
        }

    def tokenizer_function(self, dataset: DatasetDict):
        """Tokenizer function for the dataset."""
        return self.tokenizer(dataset[self.fine_tuning_config.text_column], truncation=True)

    def tokenize(self, limit=True):
        """Tokenize the input data."""
        tokenized_dataset = self.dataset.map(self.tokenizer_function, batched=True)

        if limit:
            limited_train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(1000))
            limited_eval_dataset = tokenized_dataset["test"].shuffle(seed=42).select(range(1000))
            return limited_train_dataset, limited_eval_dataset

        return tokenized_dataset["train"], tokenized_dataset["test"]
