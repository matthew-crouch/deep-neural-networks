"""Example of inference for a chat bot style model."""

from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    model = AutoModelForCausalLM.from_pretrained(model_name)

    chat_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    chat_model = HuggingFacePipeline(pipeline=chat_pipeline)

    while True:
        user = input("User: ")
        response = chat_model.invoke(user)
        print("Model: ", response)
