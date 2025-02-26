"""Tokenizer class for the pipeline."""

import logging

from datasets import DatasetDict
from transformers import (
    AutoTokenizer,
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class Tokenizer:
    """Tokenizer class for the pipeline."""

    def __init__(self, model_name: str, config: dict):
        """Initialise the tokenizer.

        :param model_name: str: The model name to use for tokenization.
        :param config: dict: The configuration for the tokenizer.
        """
        self.auto_tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.auto_tokenizer.pad_token_id is None:
            self.auto_tokenizer.pad_token = self.auto_tokenizer.eos_token
        self.config = config

    def tokenizer_function(self, dataset: DatasetDict) -> DatasetDict:
        """Tokenizer function for the dataset.

        :param dataset: DatasetDict: The dataset to tokenize.
        :return: model_inputs: The tokenized dataset.
        """
        model_inputs = self.auto_tokenizer(
            dataset[self.config.text_column],
            truncation=True,
            padding="max_length",
            max_length=5020,
        )
        with self.auto_tokenizer.as_target_tokenizer():
            labels = self.auto_tokenizer(
                dataset[self.config.target_column],
                truncation=True,
                padding="max_length",
                max_length=5020,
            )

        labels["input_ids"] = [
            [(token if token != self.auto_tokenizer.pad_token_id else -100) for token in label]
            for label in labels["input_ids"]
        ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def tokenize(self, dataset: DatasetDict, limit: bool = True) -> tuple[DatasetDict, DatasetDict]:
        """Tokenize the input data.

        :param dataset: DatasetDict: The dataset to tokenize.
        :param limit: bool: Limit the dataset size.
        :return: tokenized_dataset: The tokenized dataset.
        """
        logger.info("Tokenizing the dataset...")

        tokenized_dataset = dataset.map(self.tokenizer_function, batched=True)

        if limit:
            limited_train_dataset = (
                tokenized_dataset["train"].shuffle(seed=42).select(range(self.config.sample_size))
            )
            limited_eval_dataset = (
                tokenized_dataset["test"].shuffle(seed=42).select(range(self.config.sample_size))
            )
            return limited_train_dataset, limited_eval_dataset

        logger.info("Tokenizing completed...")

        return tokenized_dataset["train"], tokenized_dataset["test"]
