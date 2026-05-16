import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = 'cointegrated/rubert-tiny2'
MAX_LEN = 128
ASPECTS = ['качество', 'цена', 'доставка', 'сервис', 'соответствие', 'другое']


class ReviewAnalyzer(nn.Module):
    """
    Multi-task модель на базе rubert-tiny2.
    Голова 1: бинарная классификация тональности (positive/negative)
    Голова 2: multi-label классификация аспектов
    """
    def __init__(self, model_name=MODEL_NAME, num_aspects=len(ASPECTS), dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.sentiment_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, 2)
        )
        self.aspect_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, num_aspects)
        )

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0, :])
        return self.sentiment_head(cls), self.aspect_head(cls)


def load_model(weights_path, device):
    """Загружает модель из файла весов."""
    model = ReviewAnalyzer().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def predict_review(text, model, tokenizer, device, threshold=0.5):
    """
    Анализирует один отзыв.
    Возвращает тональность, уверенность и список аспектов.
    """
    encoding = tokenizer(
        str(text),
        max_length=MAX_LEN,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        sent_logits, asp_logits = model(input_ids, attention_mask)

    sent_probs = torch.softmax(sent_logits, dim=1).cpu().numpy()[0]
    asp_probs = torch.sigmoid(asp_logits).cpu().numpy()[0]

    sentiment = 'positive' if sent_probs[1] > 0.5 else 'negative'
    confidence = float(max(sent_probs))

    aspects = [
        {'aspect': ASPECTS[i], 'prob': float(asp_probs[i])}
        for i in np.argsort(asp_probs)[::-1]
        if asp_probs[i] > threshold and ASPECTS[i] != 'другое'
    ]

    return sentiment, confidence, aspects
