"""Fine-tuning pipeline for transformers models."""

import logging
from enum import Enum

import torch
from datasets import DatasetDict
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


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
        mode: TaskType,
        fine_tuning_config: dict,
    ):
        """Initialize the fine-tuning pipeline."""
        self.dataset = None
        transformers = self._mode_options.get(mode)
        transformer_model, model_name, model_kwargs = (
            transformers.get("task"),
            transformers.get("models"),
            transformers.get("model_kwargs"),
        )

        self.fine_tuning_config = FineTuningConfig(**fine_tuning_config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = transformer_model.from_pretrained(
            model_name, torch_dtype="auto", **model_kwargs
        )
        self.model.to(self.device)

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
                "data_collector": None,
                "trainer": {
                    "type": Seq2SeqTrainer,
                    "trainer_kwargs": Seq2SeqTrainingArguments(
                        "test-finetuned",
                        evaluation_strategy="epoch",
                        learning_rate=1e-5,
                        weight_decay=0.01,
                        num_train_epochs=3,
                        per_device_train_batch_size=16,
                        per_device_eval_batch_size=16,
                        fp16=True,
                    ),
                },
            },
        }

    def tokenizer_function(self, dataset: DatasetDict):
        """Tokenizer function for the dataset."""
        model_inputs = self.tokenizer(
            dataset[self.fine_tuning_config.text_column],
            truncation=True,
            padding="max_length",
            max_length=512,
        )
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                dataset[self.fine_tuning_config.target_column],
                truncation=True,
                padding="max_length",
                max_length=150,
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def tokenize(self, limit=True) -> tuple[DatasetDict, DatasetDict]:
        """Tokenize the input data."""
        logger.info("Tokenizing the dataset...")

        tokenized_dataset = self.dataset.map(self.tokenizer_function, batched=True)

        if limit:
            limited_train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(1000))
            limited_eval_dataset = tokenized_dataset["test"].shuffle(seed=42).select(range(1000))
            return limited_train_dataset, limited_eval_dataset

        logger.info("Tokenizing completed...")

        return tokenized_dataset["train"], tokenized_dataset["test"]

    def run(self, dataset: DatasetDict):
        """Fine-tune the model."""
        self.dataset = dataset

        train_data, eval_data = self.tokenize()

        trainer = self._mode_options.get(TaskType.TEXT_SUMMARISATION).get("trainer")

        auto_model = trainer.get("type")
        trainer = auto_model(
            model=self.model,
            args=trainer.get("trainer_kwargs"),
            train_dataset=train_data,
            eval_dataset=eval_data,
            tokenizer=self.tokenizer,
        )

        logger.info("Starting Fine Tuning...")
        trainer.train()
        logger.info("Fine Tuning Completed...")
        return trainer
