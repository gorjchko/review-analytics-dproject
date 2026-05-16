"""
Скрипт обучения модели
=======================
Использование:
    python train.py

Требования:
    - data/reviews_17cat.csv  — основной датасет (17 категорий)
    - data/kartaslovsent.csv  — тональный словарь KartaSlovSent

После обучения сохраняет:
    - weights/best_model.pth  — веса модели
    - data/karta_lookup.pkl   — lookup по KartaSlovSent
"""

import os
import pickle
import re
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from modules.model import ReviewAnalyzer, MODEL_NAME
from modules.preprocessing import (
    ASPECT_SEEDS, STOPWORDS, build_aspect_dict, clean_and_lemmatize, tag_aspects
)

# Пути
DATA_DIR    = 'data'
WEIGHTS_DIR = 'weights'
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

REVIEWS_PATH = os.path.join(DATA_DIR, 'reviews_17cat.csv')
KARTA_PATH   = os.path.join(DATA_DIR, 'kartaslovsent.csv')

MAX_LEN    = 128
BATCH_SIZE = 32
EPOCHS     = 4


class ReviewDataset(Dataset):
    def __init__(self, texts, sentiment_labels, aspect_labels, tokenizer):
        self.texts = texts
        self.sentiment_labels = sentiment_labels
        self.aspect_labels = aspect_labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            max_length=MAX_LEN,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'sentiment':      torch.tensor(self.sentiment_labels[idx], dtype=torch.long),
            'aspects':        torch.tensor(self.aspect_labels[idx], dtype=torch.float)
        }


def rating_to_sentiment(r):
    if r >= 4.0: return 'positive'
    if r <= 2.0: return 'negative'
    return 'neutral'


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Устройство: {device}')

    # 1. Загрузка данных
    print('\n[1/5] Загрузка данных...')
    df = pd.read_csv(REVIEWS_PATH)
    df_ru = df[df['language'] == 'russian'].copy()
    df_ru = df_ru.dropna(subset=['text']).reset_index(drop=True)
    print(f'Русских отзывов: {len(df_ru)}')

    df_karta = pd.read_csv(KARTA_PATH, sep=';')
    karta_lookup = {row['term']: {'tag': row['tag'], 'value': row['value']}
                    for _, row in df_karta.iterrows()}
    with open(os.path.join(DATA_DIR, 'karta_lookup.pkl'), 'wb') as f:
        pickle.dump(karta_lookup, f)
    print(f'KartaSlovSent: {len(karta_lookup)} слов')

    # 2. Лемматизация и разметка аспектов
    print('\n[2/5] Лемматизация и разметка аспектов...')
    df_ru['sentiment'] = df_ru['rating'].apply(rating_to_sentiment)
    df_ru['lemmas'] = df_ru['text'].apply(clean_and_lemmatize)

    # Строим словарь аспектов
    pos_counter = Counter()
    neg_counter = Counter()
    for _, row in df_ru.iterrows():
        if row['sentiment'] == 'positive':
            pos_counter.update(row['lemmas'])
        elif row['sentiment'] == 'negative':
            neg_counter.update(row['lemmas'])

    candidate_pool = set(
        [w for w, _ in (pos_counter + neg_counter).most_common(500)]
    )
    aspect_dict = build_aspect_dict(ASPECT_SEEDS, candidate_pool)
    df_ru['aspects'] = df_ru['lemmas'].apply(lambda x: tag_aspects(x, aspect_dict))

    # 3. Подготовка для обучения
    print('\n[3/5] Подготовка данных для обучения...')
    df_model = df_ru[df_ru['sentiment'].isin(['positive', 'negative'])].copy()
    df_model = df_model.reset_index(drop=True)

    SENTIMENT_MAP = {'negative': 0, 'positive': 1}
    df_model['sentiment_label'] = df_model['sentiment'].map(SENTIMENT_MAP)

    ASPECTS = ['качество', 'цена', 'доставка', 'сервис', 'соответствие', 'другое']
    mlb = MultiLabelBinarizer(classes=ASPECTS)
    aspect_labels = mlb.fit_transform(df_model['aspects'])

    with open(os.path.join(DATA_DIR, 'mlb.pkl'), 'wb') as f:
        pickle.dump(mlb, f)

    texts = df_model['text'].values
    sent_labels = df_model['sentiment_label'].values

    train_idx, temp_idx = train_test_split(
        np.arange(len(df_model)), test_size=0.2, random_state=42,
        stratify=sent_labels
    )
    val_idx, _ = train_test_split(
        temp_idx, test_size=0.5, random_state=42,
        stratify=sent_labels[temp_idx]
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = ReviewDataset(texts[train_idx], sent_labels[train_idx],
                             aspect_labels[train_idx], tokenizer)
    val_ds   = ReviewDataset(texts[val_idx],   sent_labels[val_idx],
                             aspect_labels[val_idx],   tokenizer)

    class_counts = np.bincount(sent_labels[train_idx])
    weights = 1.0 / class_counts
    sample_weights = torch.tensor([weights[l] for l in sent_labels[train_idx]])
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    # 4. Обучение
    print('\n[4/5] Обучение модели...')
    model = ReviewAnalyzer(MODEL_NAME, len(ASPECTS)).to(device)

    sent_criterion = nn.CrossEntropyLoss()
    asp_criterion  = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW([
        {'params': model.bert.parameters(),          'lr': 2e-5},
        {'params': model.sentiment_head.parameters(), 'lr': 1e-4},
        {'params': model.aspect_head.parameters(),   'lr': 1e-4},
    ], weight_decay=0.01)

    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )
    scaler = GradScaler()
    best_f1 = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f'Эпоха {epoch+1}/{EPOCHS}'):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sentiment      = batch['sentiment'].to(device)
            aspects        = batch['aspects'].to(device)

            optimizer.zero_grad()
            with autocast():
                s_logits, a_logits = model(input_ids, attention_mask)
                loss = sent_criterion(s_logits, sentiment) * 0.6 + \
                       asp_criterion(a_logits, aspects) * 0.4

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()

        # Валидация
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                with autocast():
                    s_logits, _ = model(
                        batch['input_ids'].to(device),
                        batch['attention_mask'].to(device)
                    )
                val_preds.extend(s_logits.argmax(dim=1).cpu().numpy())
                val_true.extend(batch['sentiment'].numpy())

        from sklearn.metrics import f1_score
        f1 = f1_score(val_true, val_preds, average='binary')
        print(f'  Loss: {total_loss/len(train_loader):.4f} | Sentiment F1: {f1:.4f}')

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, 'best_model.pth'))
            print('  ✓ Лучшая модель сохранена')

    # 5. Готово
    print(f'\n[5/5] Обучение завершено. Лучший F1: {best_f1:.4f}')
    print(f'Веса сохранены в {WEIGHTS_DIR}/best_model.pth')


if __name__ == '__main__':
    main()
