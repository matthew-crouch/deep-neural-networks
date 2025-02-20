import evaluate
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TextClassificationPipeline,
    AutoModelForCausalLM
)
from enum import Enum

class ModelType(Enum):

    sequence_classification = AutoModelForSequenceClassification
    text_generation = AutoModelForCausalLM

class FineTunerPipeline:
    def __init__(self, model_name: str, dataset: DatasetDict, mode: ModelType):
        self.model_name = model_name
        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = mode.from_pretrained(self.model_name)
