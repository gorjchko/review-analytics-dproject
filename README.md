# Система бизнес-аналитики отзывов

Дипломный проект. Система анализирует русскоязычные отзывы на товары и выдаёт:
- **Тональность** — позитивный / негативный (F1 = 0.98)
- **Аспекты** — качество, цена, доставка, сервис, соответствие описанию
- **Скор доверия** — оценка натуральности отзыва от 0 до 100%

## 🛠 Требования

- Python 3.10+
- pip
- GPU опционально (без GPU работает медленнее, но корректно)

## 🚀 Быстрый старт

### 1. Клонировать репозиторий

```bash
git clone https://github.com/YOUR_USERNAME/diploma.git
cd diploma
```

### 2. Установить зависимости

```bash
pip install -r requirements.txt
```

### 3. Подготовить данные

Положить в папку `data/`:
- `reviews_17cat.csv` — датасет отзывов (17 категорий товаров)
- `kartaslovsent.csv` — тональный словарь KartaSlovSent

Ссылки на датасеты: см. раздел «Данные» ниже.

### 4. Обучить модель (один раз)

```bash
python train.py
```

Обучение занимает ~5 минут на GPU, ~2-3 часа на CPU.  
После обучения в папке `weights/` появится файл `best_model.pth`.

> **Альтернатива:** если преподаватель хочет пропустить обучение,  
> скачайте готовые веса: [ссылка на Google Drive]

### 5. Запустить анализ

```bash
# Анализ CSV файла с отзывами
python main.py --input reviews.csv

# Анализ одного отзыва
python main.py --text "Отличный телефон, камера супер, доставили быстро!"

# Сохранить результаты в отдельный файл
python main.py --input reviews.csv --output my_results.csv
```

Формат входного CSV: файл должен содержать колонку `text` с текстами отзывов.

## 📁 Структура проекта

```
diploma/
├── main.py              # Точка входа — анализ отзывов
├── train.py             # Обучение модели
├── requirements.txt     # Зависимости
├── README.md
├── .gitignore
│
├── modules/
│   ├── model.py         # Архитектура rubert-tiny2 + инференс
│   ├── antispam.py      # Модуль оценки доверия
│   └── preprocessing.py # Лемматизация и разметка аспектов
│
├── data/                # Датасеты (не включены в репозиторий)
│   └── .gitkeep
│
├── weights/             # Веса модели (не включены в репозиторий)
│   └── .gitkeep
│
└── notebooks/           # Исследовательские ноутбуки Colab
    ├── 01_eda_frequency.ipynb
    ├── 02_model_training.ipynb
    ├── 03_antispam.ipynb
    └── 04_demo.ipynb
```

## 🧠 Архитектура

Базовая модель: **rubert-tiny2** (cointegrated/rubert-tiny2)  
Архитектура: multi-task — одна модель решает две задачи одновременно через две головы классификации.

Антиспам: взвешенная сумма 5 признаков (длина, лексическое разнообразие, эмоциональный баланс, качество текста, уникальность).

## 📊 Результаты

| Метрика | Значение |
|---|---|
| Sentiment F1 (positive) | 0.98 |
| Sentiment F1 (negative) | 0.56 |
| Aspect F1 (micro avg) | 0.90 |
| Accuracy | 0.96 |

## 📦 Данные

- [Reviews Dataset (17 категорий)](https://github.com/akanat/russian_reviews_dataset)
- [KartaSlovSent](https://github.com/dkulagin/kartaslov/tree/master/dataset/kartaslovsent)
