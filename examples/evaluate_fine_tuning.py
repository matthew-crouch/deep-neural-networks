"""Example of inference for a chat bot style model."""

from langchain_huggingface.llms import HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    AutoModelForSequenceClassification,
)
import pandas as pd
import torch
from peft import PeftModel

if __name__ == "__main__":
    model_name = "llama-1b-fail-message/checkpoint-13500"
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    breakpoint()
    model.load_adapter(model_name)
    breakpoint()

    data = pd.read_csv("./data/test_data_fails.csv")[["fail_message", "fail_type"]]

    classification_pipeline = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    breakpoint()
    predictions = classification_pipeline(data["fail_message"].tolist())
    breakpoint()
