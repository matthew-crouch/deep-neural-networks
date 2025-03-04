"""Model options for the pipeline."""

from enum import Enum

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
)

from src.llms.pipelines.custom_trainer import CustomTrainer


class TaskType(Enum):
    """Enum class for model types."""

    SEQUENCE_CLASSIFICATION = "sequence_classification"
    TEXT_GENERATION = "text_generation"
    TEXT_SUMMARISATION = "summarisation"
    CAUSAL_LLM = "causal_llm"


def config(**kwargs) -> dict:
    """Model options for the pipeline.

    :param kwargs: dict: The keyword arguments for the model options.
    :return: dict: The model options for the pipeline.
    """
    per_device_train_batch_size = kwargs.get("per_device_train_batch_size")
    per_device_eval_batch_size = kwargs.get("per_device_eval_batch_size")
    device = kwargs.get("device")
    ft_model_name = kwargs.get("ft_model_name")
    mode_options = {
        TaskType.TEXT_GENERATION: {
            "task": AutoModelForCausalLM,
            # "models": "meta-llama/Llama-3.2-3B-Instruct",
            # "models": "gpt2",
            "models": "meta-llama/Llama-3.2-1B",
            "use_ddp": True,
            "model_kwargs": {},
            "trainer": {
                "type": Trainer,
                "trainer_kwargs": TrainingArguments(
                    ft_model_name,
                    evaluation_strategy="epoch",
                    learning_rate=1e-5,
                    weight_decay=0.01,
                    num_train_epochs=10,
                    gradient_accumulation_steps=1,
                    per_device_train_batch_size=per_device_train_batch_size,
                    per_device_eval_batch_size=per_device_eval_batch_size,
                    fp16=(device.type == "cuda"),
                    remove_unused_columns=False,
                    # deepspeed="/home/ubuntu/deep-neural-networks/zero_stage2_config.json",
                ),
            },
        },
        TaskType.SEQUENCE_CLASSIFICATION: {
            "task": AutoModelForSequenceClassification,
            "models": "meta-llama/Llama-3.2-1B",
            # "models": "google-bert/bert-base-cased",
            "use_ddp": True,
            "model_kwargs": {},
            "trainer": {
                "type": CustomTrainer,
                "trainer_kwargs": TrainingArguments(
                    ft_model_name,
                    evaluation_strategy="epoch",
                    learning_rate=1e-5,
                    weight_decay=0.01,
                    num_train_epochs=10,
                    gradient_accumulation_steps=1,
                    per_device_train_batch_size=per_device_train_batch_size,
                    per_device_eval_batch_size=per_device_eval_batch_size,
                    fp16=(device.type == "cuda"),
                    remove_unused_columns=False,
                ),
            },
        },
        TaskType.TEXT_SUMMARISATION: {
            "task": AutoModelForSeq2SeqLM,
            "models": "google/pegasus-xsum",
            "model_kwargs": {},
            "data_collector": None,
            "use_ddp": True,
            "trainer": {
                "type": Seq2SeqTrainer,
                "trainer_kwargs": Seq2SeqTrainingArguments(
                    ft_model_name,
                    evaluation_strategy="epoch",
                    learning_rate=1e-5,
                    weight_decay=0.01,
                    num_train_epochs=3,
                    per_device_train_batch_size=per_device_train_batch_size,
                    per_device_eval_batch_size=per_device_eval_batch_size,
                    fp16=(device.type == "cuda"),
                    remove_unused_columns=False,
                ),
            },
        },
    }
    return mode_options
