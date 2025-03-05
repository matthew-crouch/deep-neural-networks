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

# from datasets import load_dataset
from peft import LoraConfig
from transformers import BitsAndBytesConfig

from data.label_mapping import LABEL_MAPPING
from src.dataset.web_scrape import convert_to_dataset_dict
from src.llms.fine_tuning_pipeline import FineTunerPipeline, TaskType
from src.llms.pipelines.custom_trainer import calculate_class_weights, compute_metrics


def fetch_fail_dataset():
    """Fetch fail dataset."""
    dataset = pd.read_csv("./data/data.csv")
    dataset = dataset[["fail_message", "fail_type"]]
    dataset = dataset.rename(columns={"fail_type": "label", "fail_message": "text"}).reset_index(
        drop=True
    )
    dataset = dataset.dropna()
    dataset = dataset[
        (dataset["label"] != "nul")
        & (dataset["label"] != "UNIMPL")
        & (dataset["label"] != "Uncategorised")
        & (dataset["label"] != "Test")
    ]
    dataset["label"] = dataset["label"].map(LABEL_MAPPING)
    return dataset.sample(20000).reset_index(drop=True)


if __name__ == "__main__":
    dataset = fetch_fail_dataset()
    dataset = convert_to_dataset_dict(dataset)

    class_weights = calculate_class_weights(dataset["train"].to_pandas())
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float16)

    ft_pipeline = FineTunerPipeline(
        mode=TaskType.SEQUENCE_CLASSIFICATION,
        fine_tuning_config={
            "ft_model_name": "llama-1b-fail-message-v2",
            "text_column": "label",
            "target_column": "text",
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "class_weights_tensor": class_weights_tensor,
            "compute_metrics": compute_metrics,
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
