"""Example of inference for a chat bot style model."""

import torch
from langchain_huggingface.llms import HuggingFacePipeline
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.2-1B"
    lora_model = "llama-1b-fail-message-generation"
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    base_model = AutoModelForCausalLM.from_pretrained(model_name)

    model = PeftModel.from_pretrained(base_model, lora_model)

    chat_pipeline = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    chat_model = HuggingFacePipeline(pipeline=chat_pipeline)

    while True:
        user = input("Summarise: ")
        response = chat_model.invoke(user)
        print("Model: ", response)
