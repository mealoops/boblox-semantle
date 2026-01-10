# =========================
# engine.py
# =========================
print("🔥 engine.py importé")

import numpy as np
import random
import gzip
from typing import Dict

# =========================
# CONSTANTS
# =========================
MAX_WORDS = 400_000
COMMON_WORDS = {
    "to", "if", "must", "and", "the", "a", "an", "of", "in", "on", "for",
    "with", "as", "by", "at", "from", "up", "down", "go", "make", "take",
    "do", "say", "come", "get", "see", "know", "want", "please", "tell"
}

# =========================
# UTILITIES
# =========================
def adjust_score(word, score):
    return score * 0.8 if word in COMMON_WORDS else score


def similarity(v1, v2):
    cosine = np.dot(v1, v2)
    euclid = 1 / (1 + np.linalg.norm(v1 - v2))
    return 0.8 * cosine + 0.2 * euclid


def temperature(rank, total):
    if rank <= 10:
        return round(100 - (rank - 1) / 10 * 10, 2)
    elif rank <= 100:
        return round(90 - (rank - 10) / 90 * 30, 2)
    elif rank <= 1000:
        return round(60 - (rank - 100) / 900 * 40, 2)
    elif rank <= 5000:
        return round(20 - (rank - 1000) / 4000 * 30, 2)
    else:
        return round(-10 - (rank - 5000) / (27086 - 5000) * 10, 2)


def temp_emoji(t):
    if t > 80: return "🌋"
    if t > 60: return "🥵"
    if t > 50: return "🔥"
    if t > 30: return "🌞"
    if t > 20: return "🌤️"
    if t > 0:  return "❄️"
    return "🧊"

# =========================
# GAME ENGINE (GLOBAL)
# =========================
class GameEngine:
    embeddings: Dict[str, np.ndarray] = {}
    secret_words: Dict[str, list] = {}

    @classmethod
    def load_glove(cls, path):
        with open(path, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= MAX_WORDS:
                    break
                parts = line.strip().split()
                word = parts[0]
                try:
                    vec = np.array(parts[1:], dtype=float)
                    vec /= np.linalg.norm(vec)
                    cls.embeddings[word] = vec
                except:
                    continue
        print(f"✅ Embeddings chargés : {len(cls.embeddings)} mots")

    @classmethod
    def load_secrets(cls, path):
        current = None
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("["):
                    current = line[1:-1]
                    cls.secret_words[current] = []
                else:
                    cls.secret_words[current].append(line.lower())
        print(f"✅ Thèmes chargés : {len(cls.secret_words)}")

# =========================
# GAME TABLE (ONE GAME)
# =========================
class GameTable:
    def __init__(self, table_id):
        self.id = table_id
        self.secret_word = None
        self.ranking = None
        self.rank_map = None
        self.finished = False

    # -------- SECRET PICKING --------
    def pick_from_glove(self):
        self.secret_word = random.choice(list(GameEngine.embeddings.keys()))
        self._build_ranking()

    def pick_from_secrets(self):
        theme = random.choice(list(GameEngine.secret_words.keys()))
        self.secret_word = random.choice(GameEngine.secret_words[theme])
        self._build_ranking()

    def pick_from_theme(self, theme):
        self.secret_word = random.choice(GameEngine.secret_words[theme])
        self._build_ranking()

    # -------- CORE LOGIC --------
    def _build_ranking(self):
        secret_vec = GameEngine.embeddings[self.secret_word]
        scores = []
        for word, vec in GameEngine.embeddings.items():
            score = adjust_score(word, similarity(secret_vec, vec))
            scores.append((word, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        self.ranking = scores
        self.rank_map = {w: i + 1 for i, (w, _) in enumerate(scores)}

    # -------- GAMEPLAY --------
    def guess(self, word):
        if self.finished:
            return {"error": "game_finished"}

        word = word.lower()
        if word not in self.rank_map:
            return {"error": "unknown_word"}

        rank = self.rank_map[word]
        temp = temperature(rank, len(self.rank_map))

        found = word == self.secret_word
        if found:
            self.finished = True

        return {
            "word": word,
            "rank": rank,
            "temperature": round(temp, 2),
            "emoji": temp_emoji(temp),
            "found": found
        }

    def hint(self, level):
        ranges = {
            1: (1000, 2000),
            2: (250, 500),
            3: (50, 100),
            4: (5, 10),
        }
        if level == 99:
            return self.secret_word

        low, high = ranges[level]
        candidates = self.ranking[low:high]
        return random.choice(candidates)[0]

# =========================
# GAME MANAGER (MULTI TABLE)
# =========================
class GameManager:
    def __init__(self):
        self.tables = {}

    def create(self, table_id):
        self.tables[table_id] = GameTable(table_id)

    def cancel(self, table_id):
        self.tables.pop(table_id, None)

    def get(self, table_id):
        return self.tables.get(table_id)





