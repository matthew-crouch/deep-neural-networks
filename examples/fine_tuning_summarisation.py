"""Example of fine-tuning a model on the XSum dataset.

This example demonstrates how we can perform fine tuning over a number
of different tasks using the `FineTunerPipeline` class. The pipeline
supports sequence classification, text generation, and summarisation tasks.

From a distributed compute standpoint there are a number of ways to exectute
this script. Firstly you can simply call this natively with python

`python examples/fine_tuning.py`

This will use Distributed Data Parallel (DDP) to train the model on all available
GPUs.

We can also leverage FullyShardedDataParllel (FSDP) by using the following command:

`accelerate launch examples/fine_tuning.py`

This will use ZeroRedundancyOptimizer (ZeRO) to train the model on all available GPUs.
In th native config we use zero_stage 2 which will shard the gardients and optimiser states
but not the raw parameters.
"""

import pandas as pd
import torch

from datasets import load_dataset
from peft import LoraConfig
from transformers import BitsAndBytesConfig

from src.dataset.web_scrape import convert_to_dataset_dict
from src.llms.fine_tuning_pipeline import FineTunerPipeline, TaskType


def fetch_fail_dataset():
    """Fetch fail dataset."""
    dataset = pd.read_csv("./data/nevis_fail_message_summarisation.csv")
    dataset = dataset[["fail_message", "comment", "fail_signature"]]
    dataset = dataset.rename(columns={"fail_message": "label", "comment": "text"}).reset_index(
        drop=True
    )
    # dataset = dataset.drop_duplicates(subset=["fail_signature"])
    dataset = dataset.dropna(subset=["label"])
    dataset = dataset.fillna("No Comment")
    return dataset.reset_index(drop=True)


if __name__ == "__main__":
    # dataset = fetch_fail_dataset()
    # dataset = convert_to_dataset_dict(dataset)
    dataset = load_dataset("cnn_dailymail", "3.0.0")

    ft_pipeline = FineTunerPipeline(
        mode=TaskType.TEXT_SUMMARISATION,
        fine_tuning_config={
            "ft_model_name": "llama-1b-fail-message-generation",
            "text_column": "article",
            "target_column": "highlights",
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 4,
            "sample_size": 1000,
            "quantisation": {
                "enabled": True,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=False,
                    bnb_4bit_quant_type="nf4",
                ),
            },
            "lora": {
                "enabled": True,
                "lora_config": LoraConfig(
                    r=8,
                    lora_alpha=32,
                    lora_dropout=0.1,
                    bias="none",
                    # target_modules=["o_proj", "qkv_proj"],
                    target_modules=["q_proj", "v_proj"],
                ),
            },
        },
    )
    _ = ft_pipeline.run(dataset=dataset)
