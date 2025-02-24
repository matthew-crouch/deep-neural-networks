"""Example of fine-tuning a model on the XSum dataset."""

from datasets import load_dataset
from peft import LoraConfig

from src.fine_tuning_pipeline import FineTunerPipeline, TaskType

if __name__ == "__main__":
    dataset = load_dataset("xsum", trust_remote_code=True)

    ft_pipeline = FineTunerPipeline(
        mode=TaskType.TEXT_SUMMARISATION,
        fine_tuning_config={
            "ft_model_name": "custom_model",
            "text_column": "document",
            "target_column": "summary",
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 2,
            "sample_size": 100,
            "lora": {
                "enabled": True,
                "lora_config": LoraConfig(
                    r=8,
                    lora_alpha=32,
                    lora_dropout=0.1,
                    bias="none",
                    target_modules=["q_proj", "v_proj"],
                ),
            },
        },
    )
    _ = ft_pipeline.run(dataset=dataset)
