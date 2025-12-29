from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random
import numpy as np
import gzip
import os
import urllib.request

app = FastAPI()

# ✅ CORS : Permettre à Roblox d'accéder à l'API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permet tous les domaines
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# MODEL POUR LES REQUÊTES
# -------------------------
class GuessRequest(BaseModel):
    mot: str

class NewGameRequest(BaseModel):
    mode: str  # "random", "hard", ou "theme_1", "theme_2", etc.

# -------------------------
# TÉLÉCHARGER GLOVE SI ABSENT
# -------------------------
GLOVE_FILE = "glove_cemantle_filtered.txt.gz"
GLOVE_URL = "https://ton-lien-google-drive-ou-autre.com/glove_cemantle_filtered.txt.gz"

if not os.path.exists(GLOVE_FILE):
    print(f"⏳ Téléchargement de {GLOVE_FILE}...")
    try:
        urllib.request.urlretrieve(GLOVE_URL, GLOVE_FILE)
        print(f"✅ {GLOVE_FILE} téléchargé!")
    except Exception as e:
        print(f"❌ Erreur téléchargement: {e}")

# -------------------------
# IMPORT DU JEU
# -------------------------
import Jeu_complet

# -------------------------
# INIT DU JEU
# -------------------------
print("⏳ Chargement des thèmes et embeddings...")
themes = Jeu_complet.load_secret_words("mots_secrets.txt")
embeddings = Jeu_complet.load_glove(GLOVE_FILE, Jeu_complet.MAX_WORDS)
print(f"✅ {len(embeddings)} mots chargés!")

# Variables globales pour la session en cours
current_secret_word = None
current_ranking = None
current_ranks = None
current_mode = None

# -------------------------
# ENDPOINT : SANTÉ DE L'API
# -------------------------
@app.get("/")
def read_root():
    return {
        "status": "online",
        "words_loaded": len(embeddings),
        "themes": list(themes.keys()) if themes else []
    }

# -------------------------
# ENDPOINT : NOUVELLE PARTIE
# -------------------------
@app.post("/new_game")
def new_game(request: NewGameRequest):
    global current_secret_word, current_ranking, current_ranks, current_mode
    
    mode = request.mode
    current_mode = mode
    
    print(f"🎮 Nouvelle partie - Mode: {mode}")
    
    # Mode Random
    if mode == "random":
        all_words = []
        for words in themes.values():
            all_words.extend(words)
        
        if not all_words:
            current_secret_word = random.choice(list(embeddings.keys()))
        else:
            current_secret_word = random.choice(all_words)
    
    # Mode Hard
    elif mode == "hard":
        current_secret_word = random.choice(list(embeddings.keys()))
    
    # Mode Thème (theme_1, theme_2, ..., theme_random)
    elif mode.startswith("theme_"):
        if mode == "theme_random":
            # Thème aléatoire
            theme_list = list(themes.keys())
            selected_theme = random.choice(theme_list)
            current_secret_word = random.choice(themes[selected_theme])
        else:
            # Thème spécifique (theme_1, theme_2, ...)
            try:
                theme_index = int(mode.split("_")[1]) - 1
                theme_list = list(themes.keys())
                
                if 0 <= theme_index < len(theme_list):
                    selected_theme = theme_list[theme_index]
                    current_secret_word = random.choice(themes[selected_theme])
                else:
                    raise HTTPException(status_code=400, detail="Thème invalide")
            except:
                raise HTTPException(status_code=400, detail="Format de thème invalide")
    
    else:
        raise HTTPException(status_code=400, detail="Mode inconnu")
    
    # Vérifier que le mot existe dans les embeddings
    if current_secret_word not in embeddings:
        current_secret_word = random.choice(list(embeddings.keys()))
    
    # Construire le ranking
    print(f"🔍 Calcul du ranking pour: {current_secret_word}")
    current_ranking = Jeu_complet.build_ranking(current_secret_word, embeddings)
    current_ranks = Jeu_complet.build_rank_dict(current_ranking)
    
    print(f"✅ Partie initialisée - Mot: {current_secret_word}")
    
    return {
        "status": "ready",
        "mode": mode,
        "message": "Partie commencée!"
    }

# -------------------------
# ENDPOINT : DEVINER UN MOT
# -------------------------
@app.post("/guess")
def guess_word(request: GuessRequest):
    global current_secret_word, current_ranking, current_ranks
    
    if not current_secret_word or not current_ranking or not current_ranks:
        raise HTTPException(status_code=400, detail="Aucune partie en cours. Appelez /new_game d'abord.")
    
    mot = request.mot.lower()
    
    if mot not in embeddings:
        return {
            "error": "Mot inconnu",
            "found": False
        }
    
    rank = current_ranks[mot]
    score = current_ranking[rank - 1][1]
    temp = Jeu_complet.temperature(rank, len(current_ranking))
    
    # Victoire ?
    victory = (rank == 1)
    
    response = {
        "mot": mot,
        "rank": rank,
        "score": float(score),
        "temperature": float(temp),
        "found": True,
        "victory": victory
    }
    
    if victory:
        response["secret_word"] = current_secret_word
    
    return response

# -------------------------
# ENDPOINT : INDICE
# -------------------------
@app.get("/hint/{level}")
def get_hint_endpoint(level: int):
    global current_ranking, current_ranks
    
    if not current_ranking or not current_ranks:
        raise HTTPException(status_code=400, detail="Aucune partie en cours")
    
    hint = Jeu_complet.get_hint(current_ranking, level, current_ranks)
    
    if hint:
        return {"hint": hint, "success": True}
    else:
        return {"hint": "Aucun indice disponible", "success": False}

# -------------------------
# ENDPOINT : TOP MOTS
# -------------------------
@app.get("/top/{count}")
def get_top(count: int = 10):
    global current_ranking
    
    if not current_ranking:
        raise HTTPException(status_code=400, detail="Aucune partie en cours")
    
    top_words = [
        {"mot": word, "score": float(score)} 
        for word, score in current_ranking[:count]
    ]
    
    return {"top": top_words}

# -------------------------
# ENDPOINT : RÉVÉLER LE MOT (Give Up)
# -------------------------
@app.get("/reveal")
def reveal_secret():
    global current_secret_word
    
    if not current_secret_word:
        raise HTTPException(status_code=400, detail="Aucune partie en cours")
    
    return {
        "secret_word": current_secret_word,
        "message": "Partie terminée"
    }
