"""
Система бизнес-аналитики отзывов
=================================
Использование:
    python main.py --input reviews.csv
    python main.py --input reviews.csv --output results.csv
    python main.py --text "Отличный телефон, камера супер!"

Формат входного CSV: файл должен содержать колонку 'text' с текстами отзывов.
"""

import argparse
import os
import pickle
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from modules.model import load_model, predict_review, MODEL_NAME
from modules.antispam import compute_trust_score, compute_similarity_scores
from modules.preprocessing import clean_and_lemmatize

# Пути к файлам
WEIGHTS_PATH = os.path.join('weights', 'best_model.pth')
KARTA_PATH   = os.path.join('data', 'karta_lookup.pkl')


def load_resources():
    """Загружает модель и вспомогательные ресурсы."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Устройство: {device}')

    if not os.path.exists(WEIGHTS_PATH):
        print(f'Ошибка: файл весов не найден: {WEIGHTS_PATH}')
        print('Запустите train.py для обучения модели.')
        sys.exit(1)

    if not os.path.exists(KARTA_PATH):
        print(f'Ошибка: файл {KARTA_PATH} не найден.')
        print('Положите kartaslovsent.csv в папку data/ и запустите train.py.')
        sys.exit(1)

    print('Загружаем модель...')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = load_model(WEIGHTS_PATH, device)

    print('Загружаем KartaSlovSent...')
    with open(KARTA_PATH, 'rb') as f:
        karta_lookup = pickle.load(f)

    return model, tokenizer, karta_lookup, device


def analyze_reviews(texts, model, tokenizer, karta_lookup, device):
    """Полный анализ списка отзывов."""
    print(f'Анализируем {len(texts)} отзывов...')

    # Предвычисляем косинусное сходство
    sim_scores = compute_similarity_scores(texts)

    results = []
    for i, text in enumerate(tqdm(texts, desc='Анализ')):
        sentiment, confidence, aspects = predict_review(text, model, tokenizer, device)
        trust, breakdown = compute_trust_score(text, karta_lookup, sim_scores[i])
        results.append({
            'text': text,
            'sentiment': sentiment,
            'confidence_pct': round(confidence * 100, 1),
            'aspects': ', '.join([a['aspect'] for a in aspects]) if aspects else '—',
            'trust_score': trust,
            'trust_length':     round(breakdown['length'] * 100, 1),
            'trust_diversity':  round(breakdown['diversity'] * 100, 1),
            'trust_emotion':    round(breakdown['emotion'] * 100, 1),
            'trust_quality':    round(breakdown['quality'] * 100, 1),
            'trust_similarity': round(breakdown['similarity'] * 100, 1),
        })

    return pd.DataFrame(results)


def print_report(df):
    """Выводит текстовый отчёт в консоль."""
    print('\n' + '=' * 60)
    print('ОТЧЁТ ПО АНАЛИЗУ ОТЗЫВОВ')
    print('=' * 60)

    total = len(df)
    pos = (df['sentiment'] == 'positive').sum()
    neg = (df['sentiment'] == 'negative').sum()
    suspicious = (df['trust_score'] < 50).sum()

    print(f'\nВсего отзывов:        {total}')
    print(f'Позитивных:           {pos} ({pos/total*100:.0f}%)')
    print(f'Негативных:           {neg} ({neg/total*100:.0f}%)')
    print(f'Подозрительных:       {suspicious} ({suspicious/total*100:.0f}%)')
    print(f'Средний скор доверия: {df["trust_score"].mean():.1f}%')

    print('\n--- Упоминания аспектов ---')
    all_aspects = [
        a.strip() for aspects in df['aspects']
        for a in aspects.split(',') if a.strip() != '—'
    ]
    for aspect, count in Counter(all_aspects).most_common():
        print(f'  {aspect}: {count}')

    if suspicious > 0:
        print('\n--- Подозрительные отзывы ---')
        susp = df[df['trust_score'] < 50].sort_values('trust_score')
        for _, row in susp.iterrows():
            print(f'  [{row["trust_score"]}%] {str(row["text"])[:80]}')

    print('\n--- Детальный разбор ---')
    for _, row in df.iterrows():
        flag = '⚠' if row['trust_score'] < 50 else '✓'
        print(f'  {flag} [{row["sentiment"]} {row["confidence_pct"]}% | '
              f'доверие {row["trust_score"]}% | {row["aspects"]}]')
        print(f'     {str(row["text"])[:80]}')


def main():
    parser = argparse.ArgumentParser(description='Система анализа отзывов')
    parser.add_argument('--input',  type=str, help='Путь к CSV файлу с отзывами')
    parser.add_argument('--output', type=str, default='results.csv',
                        help='Путь для сохранения результатов (default: results.csv)')
    parser.add_argument('--text',   type=str, help='Один отзыв для анализа прямо из командной строки')
    args = parser.parse_args()

    if not args.input and not args.text:
        parser.print_help()
        sys.exit(1)

    model, tokenizer, karta_lookup, device = load_resources()

    if args.text:
        # Режим одного отзыва
        texts = [args.text]
    else:
        # Режим CSV
        if not os.path.exists(args.input):
            print(f'Ошибка: файл {args.input} не найден.')
            sys.exit(1)
        df_input = pd.read_csv(args.input)
        if 'text' not in df_input.columns:
            print(f'Ошибка: в файле {args.input} нет колонки "text".')
            sys.exit(1)
        texts = df_input['text'].dropna().tolist()

    results_df = analyze_reviews(texts, model, tokenizer, karta_lookup, device)
    print_report(results_df)

    results_df.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f'\nРезультаты сохранены в {args.output}')


if __name__ == '__main__':
    main()
