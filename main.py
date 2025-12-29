from fastapi import FastAPI
from pydantic import BaseModel
import random
import os
import numpy as np
import Jeu_complet

app = FastAPI()

# -------------------------
# MODEL POUR LES REQUÊTES
# -------------------------
class GuessRequest(BaseModel):
    mot: str

# -------------------------
# INIT DU JEU
# -------------------------
themes = Jeu_complet.load_secret_words("mots_secrets.txt")

EMBEDDING_PATH = "glove_cemantle_filtered.txt"
MAX_WORDS = 50000

# Charger les embeddings
print("⏳ Chargement des embeddings GloVe...")
embeddings = Jeu_complet.load_glove(EMBEDDING_PATH, MAX_WORDS)
print(f"✅ {len(embeddings)} mots chargés.")

# Choisir un mot secret au hasard pour cette session
secret_word = random.choice(list(embeddings.keys()))
ranking = Jeu_complet.build_ranking(secret_word, embeddings)
ranks = Jeu_complet.build_rank_dict(ranking)

# -------------------------
# ENDPOINT DEVINER UN MOT
# -------------------------
@app.post("/guess")
def guess_word(request: GuessRequest):
    mot = request.mot.lower()
    if mot not in embeddings:
        return {"error": "Mot inconnu"}
    
    rank = ranks[mot]
    score = ranking[rank - 1][1]
    temp = Jeu_complet.temperature(rank, len(ranking))
    emoji = Jeu_complet.get_temperature_emoji(temp)
    
    return {
        "mot": mot,
        "rank": rank,
        "score": score,
        "temperature": temp,
        "emoji": emoji
    }

# -------------------------
# ENDPOINT POUR UN INDICE
# -------------------------
@app.get("/hint/{level}")
def get_hint(level: int):
    hint = Jeu_complet.get_hint(ranking, level, ranks)
    return {"hint": hint if hint else "Aucun indice disponible à ce niveau"}

# -------------------------
# ENDPOINT POUR LE TOP 10
# -------------------------
@app.get("/top")
def get_top():
    top_10 = [{"mot": word, "score": score} for word, score in ranking[:10]]
    return {"top": top_10}
