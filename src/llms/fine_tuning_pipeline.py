"""Fine-tuning pipeline for transformers models."""

import logging
from typing import Any

import torch
from datasets import DatasetDict
from peft import LoraConfig, get_peft_model
from pydantic import BaseModel

from src.llms.pipelines.mode_config import TaskType, config
from src.llms.pipelines.tokenizer import Tokenizer

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class FineTuningConfig(BaseModel):
    """Dataclass for fine-tuning configuration."""

    ft_model_name: str
    text_column: str
    target_column: str
    lora: dict = {"enabled": False, "lora_config": LoraConfig()}
    quantisation: dict = {"enabled": False, "quantisation_config": LoraConfig()}
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    sample_size: int = 10
    save_model: bool = True
    class_weights_tensor: torch.tensor = None
    compute_metrics: Any = None

    class Config:
        """dataclass config."""

        arbitrary_types_allowed = True


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
        self.mode = mode
        self.fine_tuning_config = FineTuningConfig(**fine_tuning_config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._initialise_model_setup()
        model_size, num_parameters = self.get_model_size_bytes(self.model)

        self.custom_kwargs = {}
        if self.fine_tuning_config.class_weights_tensor is not None:
            self.custom_kwargs["class_weights"] = self.fine_tuning_config.class_weights_tensor.to(
                self.device
            )

        if self.fine_tuning_config.compute_metrics is not None:
            self.custom_kwargs["compute_metrics"] = self.fine_tuning_config.compute_metrics

        logger.info(f"Total Model Size in RAM: {round(model_size * 1e-9, 4)} GB")
        logger.info(f"Number of Parameters: {round(num_parameters * 1e-9, 4)} B")

    def _initialise_model_setup(self):
        """Initialise the model setup."""
        transformers = self._mode_options(self.mode)
        transformer_model, model_name, model_kwargs = (
            transformers.get("task"),
            transformers.get("models"),
            transformers.get("model_kwargs"),
        )

        if self.fine_tuning_config.quantisation.get(
            "enabled"
        ) and not self.fine_tuning_config.lora.get("enabled"):
            raise ValueError(
                "You must add Trainable Adapters since you "
                "cannot perform fine-tuning on purely quantised models"
            )

        if self.fine_tuning_config.class_weights_tensor is not None:
            model_kwargs["num_labels"] = len(self.fine_tuning_config.class_weights_tensor)

        quant_config = None
        if self.fine_tuning_config.quantisation.get("enabled"):
            quant_config = self.fine_tuning_config.quantisation.get("quantization_config")

        model = transformer_model.from_pretrained(
            model_name,
            quantization_config=quant_config,
            **model_kwargs,
        )

        if self.fine_tuning_config.lora.get("enabled"):
            model = get_peft_model(model, self.fine_tuning_config.lora.get("lora_config"))

        return model

    def get_model_size_bytes(self, model: torch.nn.Module) -> tuple:
        """Calculate the model size.

        Calculate total number of bytes occupied by model's params + buffers
        and return the number of parameters in the model.
        """
        param_size = 0
        for param in model.parameters():
            param_size += param.numel() * param.element_size()

        buffer_size = 0
        for buf in model.buffers():
            buffer_size += buf.numel() * buf.element_size()

        return (param_size + buffer_size, self.model.num_parameters())

    def _mode_options(self, mode: TaskType) -> dict:
        """Model options for the pipeline.

        :param mode: TaskType
        :return: dict
        """
        mode_options = config(
            ft_model_name=self.fine_tuning_config.ft_model_name,
            per_device_train_batch_size=self.fine_tuning_config.per_device_train_batch_size,
            per_device_eval_batch_size=self.fine_tuning_config.per_device_eval_batch_size,
            device=self.device,
        )
        if mode not in mode_options:
            raise ValueError(f"Unsupported mode: {mode}")
        return mode_options.get(mode)

    def run(self, dataset: DatasetDict):
        """Fine-tune the model."""
        trainer = self._mode_options(mode=self.mode)

        tokenizer = Tokenizer(trainer["models"], config=self.fine_tuning_config)
        train_data, eval_data = tokenizer.tokenize(dataset=dataset, limit=False)

        auto_model = trainer.get("trainer").get("type")
        # data_collator = DataCollatorForSeq2Seq(
        #     tokenizer.auto_tokenizer, model=self.model, padding=True
        # )
        trainer = auto_model(
            model=self.model,
            args=trainer.get("trainer").get("trainer_kwargs"),
            train_dataset=train_data.remove_columns(["id"]),
            eval_dataset=eval_data.remove_columns(["id"]),
            # data_collator=data_collator,
            **self.custom_kwargs,
        )

        logger.info("Starting Fine Tuning...")
        logger.info(f"Training Args: {trainer.args}")
        trainer.train()
        logger.info("Fine Tuning Completed...")

        if self.fine_tuning_config.save_model:
            self.model.save_pretrained(self.fine_tuning_config.ft_model_name)
            logger.info(f"Model saved to {self.fine_tuning_config.ft_model_name}")

            trainer.save_model(self.fine_tuning_config.ft_model_name)
            logger.info(f"Trainer saved to {self.fine_tuning_config.ft_model_name}")

        return trainer
