"""Model options for the pipeline."""

from enum import Enum

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


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
        TaskType.SEQUENCE_CLASSIFICATION: {
            "task": AutoModelForSequenceClassification,
            "models": "bert",
            "model_kwargs": {"num_labels": 2},
        },
        TaskType.TEXT_GENERATION: {
            "task": AutoModelForCausalLM,
            "models": "gpt2",
            "model_kwargs": {},
        },
        # TaskType.CAUSAL_LLM: {
        #     "task": AutoModelForCausalLM,
        #     # "models": "microsoft/Phi-3.5-mini-instruct",
        #     "models": "gpt2",
        #     "model_kwargs": {"device_map": "balanced"},
        #     # "model_kwargs": {},
        #     "data_collector": None,
        #     "trainer": {
        #         "type": Trainer,
        #         "trainer_kwargs": TrainingArguments(
        #             "test-finetuned",
        #             evaluation_strategy="epoch",
        #             learning_rate=1e-5,
        #             weight_decay=0.01,
        #             num_train_epochs=3,
        #             gradient_accumulation_steps=8,
        #             per_device_train_batch_size=per_device_train_batch_size,
        #             per_device_eval_batch_size=per_device_eval_batch_size,
        #             fp16=(device.type == "cuda"),
        #         ),
        #     },
        # },
        TaskType.TEXT_SUMMARISATION: {
            "task": AutoModelForSeq2SeqLM,
            "models": "google/pegasus-xsum",
            "model_kwargs": {"device_map": "auto"},
            "data_collector": None,
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
