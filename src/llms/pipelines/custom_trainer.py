"""Custom Trainer for Fail Classification."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    Trainer,
)


def compute_metrics(pred):
    """Compute metrics to evaluate fine-tuning."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def calculate_class_weights(dataset: pd.DataFrame):
    """Calculate the class weights for the fail types."""
    class_counts = pd.DataFrame(dataset["label"]).value_counts()
    total_counts = dataset.shape[0]
    class_weights = total_counts / (len(class_counts) * class_counts)

    class_weights = class_weights / np.sum(class_weights)
    return class_weights


class WeightedBCEWithLogitsLoss(nn.Module):
    """Weighted Binary Cross Entroy Logit Loss."""

    def __init__(self, weights):
        """Initialise the WeightedBCE class."""
        super().__init__()
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def forward(self, logits, targets):
        """Forward Pass."""
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.weights
        )
        return loss


class CustomTrainer(Trainer):
    """Custom trainer for multi-class classification."""

    def __init__(self, class_weights, *args, **kwargs):
        """Initialise the class."""
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        """How the loss is computed by Trainer.

        By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        outputs = model(**inputs)
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.

            logits = outputs["logits"]
            criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights)
            loss = criterion(logits, inputs["labels"])

        return (loss, outputs) if return_outputs else loss
