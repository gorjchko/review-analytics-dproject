import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

WEIGHTS = {
    'length':     0.15,
    'diversity':  0.25,
    'emotion':    0.25,
    'quality':    0.20,
    'similarity': 0.15
}


def score_length(text, min_words=5, max_words=300):
    """Оценивает длину отзыва. Слишком короткие и длинные подозрительны."""
    if not isinstance(text, str):
        return 0.0
    n = len(text.split())
    if n < min_words:
        return n / min_words
    if n > max_words:
        return max(0, 1 - (n - max_words) / max_words)
    return 1.0


def score_lexical_diversity(text):
    """
    Type-Token Ratio (TTR) — отношение уникальных слов к общему числу.
    Боты пишут однообразно (низкий TTR) или слишком идеально (TTR=1).
    """
    if not isinstance(text, str):
        return 0.0
    words = re.sub(r'[^а-яёА-ЯЁa-zA-Z\s]', '', text.lower()).split()
    if len(words) < 3:
        return 0.0
    if len(words) <= 20:
        return len(set(words)) / len(words)
    window = 20
    ttrs = [
        len(set(words[i:i + window])) / window
        for i in range(0, len(words) - window + 1, window // 2)
    ]
    return float(np.mean(ttrs))


def score_emotional_balance(text, karta_lookup):
    """
    Реальные отзывы содержат смешанные эмоции.
    Полностью однотонные тексты подозрительны.
    """
    if not isinstance(text, str):
        return 0.5
    words = re.sub(r'[^а-яёА-ЯЁ\s]', '', text.lower()).split()
    values = [karta_lookup[w]['value'] for w in words if w in karta_lookup]
    if len(values) < 3:
        return 0.5
    mean_val = np.mean(values)
    std_val = np.std(values)
    extremism = abs(mean_val)
    diversity_bonus = min(std_val, 0.5) / 0.5
    return float(np.clip(1.0 - extremism * (1.0 - diversity_bonus), 0, 1))


def score_text_quality(text):
    """
    Проверяет спам-паттерны: капслок, повторы, ссылки, мусорные символы.
    """
    if not isinstance(text, str) or len(text) < 2:
        return 0.0
    letters = len(re.findall(r'[а-яёА-ЯЁa-zA-Z]', text))
    letter_ratio = letters / len(text)
    repeats = len(re.findall(r'(.)\1{3,}', text))
    has_links = bool(re.search(r'http|www\.|@', text))
    caps_ratio = len(re.findall(r'[А-ЯЁA-Z]', text)) / max(letters, 1)
    penalties = [
        min(letter_ratio / 0.6, 1.0),
        max(0, 1 - repeats * 0.2),
        0.3 if has_links else 1.0,
        max(0, 1 - max(0, caps_ratio - 0.4) * 3)
    ]
    return float(np.mean(penalties))


def compute_similarity_scores(texts, threshold=0.85):
    """Вычисляет максимальное косинусное сходство каждого отзыва с остальными."""
    if len(texts) < 2:
        return np.ones(len(texts))
    vec = TfidfVectorizer(max_features=2000)
    try:
        tfidf = vec.fit_transform(texts)
        sims = cosine_similarity(tfidf).astype(np.float32)
        np.fill_diagonal(sims, 0)
        max_sims = sims.max(axis=1)
        scores = np.where(
            max_sims > threshold,
            1 - (max_sims - threshold) / (1 - threshold),
            1.0
        )
        return np.clip(scores, 0, 1)
    except Exception:
        return np.ones(len(texts))


def compute_trust_score(text, karta_lookup, sim_score=1.0):
    """
    Итоговый скор доверия к отзыву (0–100%).
    Взвешенная сумма пяти признаков.
    """
    scores = {
        'length':     score_length(text),
        'diversity':  score_lexical_diversity(text),
        'emotion':    score_emotional_balance(text, karta_lookup),
        'quality':    score_text_quality(text),
        'similarity': float(sim_score)
    }
    trust = sum(scores[k] * WEIGHTS[k] for k in scores)
    return round(trust * 100, 1), scores
