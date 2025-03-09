"""Tokenizer class for the pipeline."""

import logging

import torch
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
        # Prompt engineer for summarization since llmama is decoder only
        inputs = [f"Summarize: {doc}" for doc in dataset[self.config.text_column]]
        targets = dataset[self.config.target_column]

        model_inputs = self.auto_tokenizer(
            inputs,
            truncation=True,
            padding="max_length",
            max_length=1024,
        )
        with self.auto_tokenizer.as_target_tokenizer():
            labels = self.auto_tokenizer(
                targets,
                truncation=True,
                padding="max_length",
                max_length=128,
            )

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    def decoder_only_preprocessing(self, examples):
        """Preprocess the data for decoder only models."""
        input_ids_list = []
        labels_list = []
        max_input_length = 1024  # total sequence length
        for article, summary in zip(examples["article"], examples["highlights"]):
            # Since LLama is decoder only, we need to add a prompt to the input
            prompt = f"Summarize: {article}"

            full_text = prompt + "\n" + summary + self.auto_tokenizer.eos_token

            # Tokenize the full text sequence
            tokenized = self.auto_tokenizer(
                full_text,
                truncation=True,
                max_length=max_input_length,
                padding="max_length",
            )
            input_ids = tokenized["input_ids"]

            prompt_tokenized = self.auto_tokenizer(
                prompt + "\n",
                truncation=True,
                max_length=max_input_length,
            )
            prompt_length = len(prompt_tokenized["input_ids"])

            # Create labels: mask prompt tokens with -100 and keep summary tokens
            labels = [-100] * prompt_length + input_ids[prompt_length:]

            labels = labels[:max_input_length]
            if len(labels) < max_input_length:
                labels = labels + [-100] * (max_input_length - len(labels))

            input_ids_list.append(input_ids)
            labels_list.append(labels)
            
        return {"input_ids": input_ids_list, "labels": labels_list}

    def mulit_class_tokenizer(self, dataset):
        """Add multi-class tokenizer."""
        return self.auto_tokenizer(
            dataset["text"], padding="max_length", truncation=True, max_length=256
        )

    def tokenize(self, dataset: DatasetDict, limit: bool = True) -> tuple[DatasetDict, DatasetDict]:
        """Tokenize the input data.

        :param dataset: DatasetDict: The dataset to tokenize.
        :param limit: bool: Limit the dataset size.
        :return: tokenized_dataset: The tokenized dataset.
        """
        logger.info("Tokenizing the dataset...")

        ## TODO: Add custom tokenizer logic here
        tokenized_dataset = dataset.map(self.decoder_only_preprocessing, batched=True)
        # tokenized_dataset = dataset.map(self.mulit_class_tokenizer, batched=True)

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
