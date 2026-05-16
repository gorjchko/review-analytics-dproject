# Система бизнес-аналитики отзывов

**Студент:** Горючко Максим Сергеевич, группа ИИ-221  
**Дипломный проект**

Система анализирует русскоязычные отзывы на товары и выдаёт:
- **Тональность** — позитивный / негативный (F1 = 0.98)
- **Аспекты** — качество, цена, доставка, сервис, соответствие описанию
- **Скор доверия** — оценка натуральности отзыва от 0 до 100%

##  Требования

- Python 3.10+
- pip
- GPU опционально (без GPU работает корректно, но медленнее)

##  Быстрый старт

### 1. Клонировать репозиторий

```bash
git clone https://github.com/gorjchko/review-analytics-dproject.git
cd review-analytics-dproject
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

### 4. Получить веса модели

**Вариант А — скачать готовые веса (рекомендуется):**

Скачать `best_model.pth` по ссылке и положить в папку `weights/`:  
https://drive.google.com/file/d/1rwUL8ft9I-UNYbIZrbu4NIwLkpNHYojs/view?usp=sharing

**Вариант Б — обучить самостоятельно:**

```bash
python train.py
```

Обучение занимает ~5 минут на GPU, ~2-3 часа на CPU.

### 5. Запустить

**Графический интерфейс:**
```bash
python gui.py
```

**Командная строка:**
```bash
# Один отзыв
python main.py --text "Отличный телефон, камера супер, доставили быстро!"

# CSV файл с отзывами
python main.py --input reviews.csv

# С сохранением результатов
python main.py --input reviews.csv --output my_results.csv
```

Формат входного CSV: файл должен содержать колонку `text` с текстами отзывов.

##  Структура проекта

```
review-analytics-dproject/
├── main.py              # Точка входа — анализ через командную строку
├── train.py             # Обучение модели
├── gui.py               # Графический интерфейс (tkinter)
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
│   └── .gitkeep         # Скачать: см. шаг 4
│
└── notebooks/           # Исследовательские ноутбуки Google Colab
    ├── 01_eda_frequency.ipynb
    ├── 02_model_training.ipynb
    ├── 03_antispam.ipynb
    └── 04_demo.ipynb
```

##  Архитектура

Базовая модель: **rubert-tiny2** (cointegrated/rubert-tiny2)  
Архитектура: multi-task — одна модель решает две задачи одновременно через две головы классификации поверх общего BERT-энкодера.

Антиспам: взвешенная сумма 5 признаков (длина, лексическое разнообразие, эмоциональный баланс, качество текста, уникальность в выборке).

##  Результаты

| Метрика | Значение |
|---|---|
| Sentiment F1 (positive) | 0.98 |
| Sentiment F1 (negative) | 0.56 |
| Aspect F1 (micro avg) | 0.90 |
| Accuracy | 0.96 |

Дисбаланс F1 между классами обусловлен естественным перекосом датасета (94% позитивных отзывов против 6% негативных) — следствие эффекта Матфея в рекомендательных системах маркетплейсов. Для компенсации применён WeightedRandomSampler.

##  Данные

- [Reviews Dataset (17 категорий)](https://github.com/akanat/russian_reviews_dataset)
- [KartaSlovSent](https://github.com/dkulagin/kartaslov/tree/master/dataset/kartaslovsent)
