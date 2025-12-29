import numpy as np
import random
from collections import defaultdict

# =========================
# PARAMETERS
# =========================
EMBEDDING_PATH = "glove_cemantle_filtered.txt"
SECRET_WORDS_PATH = "mots_secrets.txt"
MAX_WORDS = 50000

# =========================
# COMMON WORDS (pénalisées)
# =========================
COMMON_WORDS = {
    "to", "if", "must", "and", "the", "a", "an", "of", "in", "on", "for",
    "with", "as", "by", "at", "from", "up", "down", "go", "make", "take",
    "do", "say", "come", "get", "see", "know", "want", "please", "tell"
}

def adjust_score(word, score):
    if word.lower() in COMMON_WORDS:
        return score * 0.8
    return score

# =========================
# LOAD SECRET WORDS BY THEME
# =========================
def load_secret_words(path):
    themes = {}
    current_theme = None
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                
                if not line or line.startswith("#"):
                    continue
                
                if line.startswith("[") and line.endswith("]"):
                    current_theme = line[1:-1]
                    themes[current_theme] = []
                
                elif current_theme:
                    themes[current_theme].append(line.lower())
        
        print(f"✅ {len(themes)} thèmes chargés:")
        for theme, words in themes.items():
            print(f"   • {theme}: {len(words)} mots")
        
        return themes
    
    except FileNotFoundError:
        print(f"❌ Fichier '{path}' introuvable!")
        return None

# =========================
# LOAD GLOVE EMBEDDINGS (txt)
# =========================
def load_glove(path, max_words):
    embeddings = {}
    
    # Détecter si c'est un fichier .gz
    if path.endswith('.gz'):
        import gzip
        open_func = lambda p: gzip.open(p, 'rt', encoding='utf-8')
    else:
        open_func = lambda p: open(p, 'r', encoding='utf-8')
    
    with open_func(path) as f:
        for i, line in enumerate(f):
            if i >= max_words:
                break
            values = line.strip().split()
            if len(values) < 2:
                continue
            word = values[0]
            try:
                vector = np.array(values[1:], dtype=float)
                norm = np.linalg.norm(vector)
                if norm == 0:
                    continue
                vector /= norm
                embeddings[word] = vector
            except:
                continue
    
    return embeddings
# =========================
# SIMILARITY CALCULATION
# =========================
def vector_similarity(v1, v2):
    cosine = np.dot(v1, v2)
    euclidean_dist = np.linalg.norm(v1 - v2)
    euclidean_sim = 1 / (1 + euclidean_dist)
    total_score = 0.8 * cosine + 0.2 * euclidean_sim
    return total_score

# =========================
# BUILD RANKING
# =========================
def build_ranking(secret_word, embeddings):
    secret_vec = embeddings[secret_word]
    scores = []
    
    for word, vec in embeddings.items():
        score = vector_similarity(secret_vec, vec)
        score = adjust_score(word, score)
        scores.append((word, score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

def build_rank_dict(ranking):
    return {word: rank + 1 for rank, (word, _) in enumerate(ranking)}

# =========================
# TEMPERATURE
# =========================
def temperature(rank, total_words):
    if rank <= 10:
        return round(100 - (rank - 1) / 10 * 10, 2)
    elif rank <= 100:
        return round(90 - (rank - 10) / 90 * 30, 2)
    elif rank <= 1000:
        return round(60 - (rank - 100) / 900 * 40, 2)
    elif rank <= 5000:
        return round(20 - (rank - 1000) / 4000 * 30, 2)
    else:
        return round(-10 - (rank - 5000) / (total_words - 5000) * 10, 2)

def get_temperature_emoji(temp):
    if temp > 80:
        return "🔥"
    elif temp > 50:
        return "🌞"
    elif temp > 20:
        return "🌤️"
    elif temp > 0:
        return "❄️"
    else:
        return "🥶"

# =========================
# HINT SYSTEM
# =========================
def get_hint(ranking, level, ranks):
    if level == 1:
        if len(ranking) > 5000:
            choices = ranking[3000:5000]
            hint_word, _ = random.choice(choices)
            return f"💡 Mot éloigné: '{hint_word}' (rang {ranks[hint_word]})"
    elif level == 2:
        if len(ranking) > 1000:
            choices = ranking[500:1000]
            hint_word, _ = random.choice(choices)
            return f"💡 Mot moyennement proche: '{hint_word}' (rang {ranks[hint_word]})"
    elif level == 3:
        if len(ranking) > 500:
            choices = ranking[100:500]
            hint_word, _ = random.choice(choices)
            return f"💡 Mot proche: '{hint_word}' (rang {ranks[hint_word]})"
    elif level == 4:
        if len(ranking) > 20:
            choices = ranking[2:20]
            hint_word, _ = random.choice(choices)
            return f"💡 TRÈS PROCHE: '{hint_word}' (rang {ranks[hint_word]})"
    return None

