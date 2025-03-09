"""Example of inference for a chat bot style model."""

import pandas as pd
import torch
from peft import PeftModel
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

from data.label_mapping import REVERSE_LABEL_MAPPING

if __name__ == "__main__":
    base_model_name = "meta-llama/Llama-3.2-1B"
    lora_model = "llama-1b-fail-message"

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name, ignore_mismatched_sizes=True
    )

    model = PeftModel.from_pretrained(base_model, lora_model)

    data = (
        pd.read_csv("./data/test_data_fails.csv")[["fail_message", "fail_type"]]
        .sample(1000)
        .reset_index(drop=True)
    )

    classification_pipeline = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    predictions = classification_pipeline(data["fail_message"].tolist())
    predictions = pd.DataFrame(predictions)
    predictions["label"] = predictions["label"].map(REVERSE_LABEL_MAPPING)

    data = pd.concat([data, predictions], axis=1)
    breakpoint()
