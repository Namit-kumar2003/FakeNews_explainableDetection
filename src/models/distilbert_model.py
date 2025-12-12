import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch.nn as nn


class FakeNewsClassifier:
    def __init__(self, model_name="distilbert-base-uncased", num_labels=2, dropout_rate=0.3):

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )

        # -------------------
        # Freeze DistilBERT layers
        # -------------------
        for param in self.model.distilbert.parameters():
            param.requires_grad = False

        # -------------------
        # Add Strong Dropout
        # -------------------
        self.model.dropout = nn.Dropout(dropout_rate)

    def tokenize(self, texts, max_length=64):  # REDUCED max_length
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
