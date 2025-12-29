from fastapi import FastAPI
from pydantic import BaseModel
import random
import numpy as np
import os
from collections import defaultdict
import Jeu_complet  # ton script existant

app = FastAPI()

# -------------------------
# MODEL POUR LES REQUÊTES
# -------------------------
class GuessRequest(BaseModel):
    mot: str

# -------------------------
# INIT DU JEU
# -------------------------
# Charger les fichiers et embeddings une seule fois au démarrage
themes = Jeu_complet.load_secret_words("mots_secrets.txt")
import gdown
import gzip

GDRIVE_ID = "1AbCdefGHIjklMNopQrsTUvWxYz"  # remplace par ton ID
LOCAL_FILE = "glove_cemantle_filtered.txt.gz"

# Téléchargement si le fichier n'existe pas déjà
if not os.path.exists(LOCAL_FILE):
    url = f"https://drive.google.com/uc?id=1xBJun3ZRx7y25YMZWCve6t6hTEosnG4u"
    gdown.download(url, LOCAL_FILE, quiet=False)

# Ouverture du fichier gzip
with gzip.open(LOCAL_FILE, "rt", encoding="utf-8") as f:
    embeddings = {}
    for i, line in enumerate(f):
        if i >= Jeu_complet.MAX_WORDS:
            break
        word, *vector = line.strip().split()
        embeddings[word] = [float(x) for x in vector]

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
    
    response = {
        "mot": mot,
        "rank": rank,
        "score": score,
        "temperature": temp,
        "emoji": emoji
    }
    return response

# -------------------------
# ENDPOINT POUR UN INDICE
# -------------------------
@app.get("/hint/{level}")
def get_hint(level: int):
    hint = Jeu_complet.get_hint(ranking, level, ranks)
    if hint:
        return {"hint": hint}
    else:
        return {"hint": "Aucun indice disponible à ce niveau"}

# -------------------------
# ENDPOINT POUR LE TOP 10
# -------------------------
@app.get("/top")
def get_top():
    top_10 = [{"mot": word, "score": score} for word, score in ranking[:10]]
    return {"top": top_10}




