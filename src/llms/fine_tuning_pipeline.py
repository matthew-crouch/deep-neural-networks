"""Fine-tuning pipeline for transformers models."""

import logging

import torch
import torch.distributed as dist
from datasets import DatasetDict
from peft import LoraConfig, get_peft_model
from pydantic import BaseModel
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel

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
    save_model: bool = False


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
        transformers = self._mode_options(mode)
        transformer_model, model_name, model_kwargs = (
            transformers.get("task"),
            transformers.get("models"),
            transformers.get("model_kwargs"),
        )

        if self.fine_tuning_config.quantisation.get(
            "load_in_4bit"
        ) and not self.fine_tuning_config.lora.get("enabled"):
            raise ValueError(
                "You must add Trainable Adapters since you"
                "cannot perform fine-tuning on purely quantised models"
            )

        model = transformer_model.from_pretrained(
            model_name,
            # torch_dtype="auto",
            quantization_config=self.fine_tuning_config.quantisation.get("quantization_config"),
            **model_kwargs,
        )

        if self.fine_tuning_config.lora.get("enabled"):
            model = get_peft_model(model, self.fine_tuning_config.lora.get("lora_config"))

        self.model = model
        logger.info(f"Total Model Size in RAM: {self.get_model_size_bytes(self.model) * 1e-9} GB")
        logger.info(f"Number of Parameters: {self.model.num_parameters() * 1e-9} B")

    def distribute_to_devices(self, trainer: dict):
        """Distribute the model to multiple GPUs."""
        if torch.cuda.device_count() > 1:
            torch.cuda.empty_cache()
            logger.info(f"Multiple GPUs found. Training on all {torch.cuda.device_count()} GPUs.")

            if trainer.get("use_ddp"):
                # This must be called and run with the accelerate framework
                self.model = DistributedDataParallel(self.model)
        else:
            self.model = DataParallel(self.model)

    def get_model_size_bytes(self, model: torch.nn.Module) -> int:
        """Calculate the total number of bytes occupied by model's params + buffers."""
        param_size = 0
        for param in model.parameters():
            param_size += param.numel() * param.element_size()

        buffer_size = 0
        for buf in model.buffers():
            buffer_size += buf.numel() * buf.element_size()

        return param_size + buffer_size

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
        train_data, eval_data = tokenizer.tokenize(dataset=dataset)

        # self.distribute_to_devices(trainer)

        auto_model = trainer.get("trainer").get("type")
        trainer = auto_model(
            model=self.model,
            args=trainer.get("trainer").get("trainer_kwargs"),
            ## Generalise this to support other tasks
            train_dataset=train_data.remove_columns(["document", "summary", "id"]),
            eval_dataset=eval_data.remove_columns(["document", "summary", "id"]),
        )
        logger.info("Starting Fine Tuning...")
        trainer.train()
        logger.info("Fine Tuning Completed...")

        if self.fine_tuning_config.save_model:
            trainer.save_model()

        dist.destroy_process_group()

        return trainer
