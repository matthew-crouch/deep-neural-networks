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

import torch
# from datasets import load_dataset
from peft import LoraConfig
from transformers import BitsAndBytesConfig

from src.dataset.web_scrape import convert_to_dataset_dict
from src.llms.fine_tuning_pipeline import FineTunerPipeline, TaskType

if __name__ == "__main__":
    # dataset = load_dataset("xsum", trust_remote_code=True)
    warhammer_sources = [
        "https://warhammer40k.fandom.com/wiki/Chaos",
        "https://warhammer40k.fandom.com/wiki/Space_Marines",
        "https://warhammer40k.fandom.com/wiki/Tau",
        "https://warhammer40k.fandom.com/wiki/Eldar",
        "https://warhammer40k.fandom.com/wiki/Necrons",
        "https://warhammer40k.fandom.com/wiki/Orks",
        "https://warhammer40k.fandom.com/wiki/Tyranids",
        "https://warhammer40k.fandom.com/wiki/Imperium_of_Man",
        "https://warhammer40k.fandom.com/wiki/Adeptus_Astartes",
        "https://warhammer40k.fandom.com/wiki/Imperial_Guard",
        "https://warhammer40k.fandom.com/wiki/Imperial_Navy",
        "https://warhammer40k.fandom.com/wiki/Inquisition",
        "https://warhammer40k.fandom.com/wiki/Adeptus_Mechanicus",
        "https://warhammer40k.fandom.com/wiki/Astra_Telepathica",
        "https://warhammer40k.fandom.com/wiki/Adeptus_Arbites",
        "https://warhammer40k.fandom.com/wiki/Adepta_Sororitas",
        "https://warhammer40k.fandom.com/wiki/Horus_Heresy",
        "https://warhammer40k.fandom.com/wiki/Great_Crusade",
    ]
    dataset, _ = convert_to_dataset_dict(sources=warhammer_sources)

    ft_pipeline = FineTunerPipeline(
        mode=TaskType.TEXT_GENERATION,
        fine_tuning_config={
            "ft_model_name": "warhammer_model",
            "text_column": "document",
            "target_column": "summary",
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            # "sample_size": len(dataset["train"]),
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

