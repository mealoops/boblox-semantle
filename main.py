from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random
import numpy as np
import gzip
import os
import urllib.request
from typing import Optional, Dict
import uuid

app = FastAPI(title="Cemantix API", version="1.0.0")

# ✅ CORS : Permettre à Roblox d'accéder
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# MODELS
# -------------------------
class NewGameRequest(BaseModel):
    mode: str  # "theme_1" à "theme_19", "theme_random", "random", "hard"
    table_id: str  # ID de la table Roblox (ex: "table1")

class GuessRequest(BaseModel):
    table_id: str
    mot: str

class HintRequest(BaseModel):
    table_id: str
    level: int  # 1-4

# -------------------------
# TÉLÉCHARGER GLOVE
# -------------------------
GLOVE_FILE = "glove_cemantle_filtered.txt.gz"
GLOVE_URL = "https://drive.usercontent.google.com/download?id=1xBJun3ZRx7y25YMZWCve6t6hTEosnG4u&export=download&authuser=1&confirm=t&uuid=fb049900-bef1-4993-aeda-497c5f148eef&at=ANTm3czvDAIkeFl7cliMlxz8oEva:1766971120225"

if not os.path.exists(GLOVE_FILE):
    print(f"⏳ Téléchargement de {GLOVE_FILE}...")
    try:
        urllib.request.urlretrieve(GLOVE_URL, GLOVE_FILE)
        print(f"✅ {GLOVE_FILE} téléchargé!")
    except Exception as e:
        print(f"⚠️ Téléchargement échoué: {e}")

# -------------------------
# IMPORT FONCTIONS DU JEU
# -------------------------
import Jeu_complet

# -------------------------
# CHARGEMENT INITIAL
# -------------------------
print("⏳ Chargement des thèmes et embeddings...")
themes = Jeu_complet.load_secret_words("mots_secrets.txt")
embeddings = Jeu_complet.load_glove(GLOVE_FILE, Jeu_complet.MAX_WORDS)
print(f"✅ {len(embeddings)} mots chargés!")

# Liste des thèmes pour référence
THEME_LIST = list(themes.keys()) if themes else []

# -------------------------
# STOCKAGE DES SESSIONS
# -------------------------
# sessions[table_id] = {
#     "secret_word": str,
#     "ranking": list,
#     "ranks": dict,
#     "mode": str,
#     "attempts": int,
#     "guesses": list
# }
sessions: Dict[str, dict] = {}

# -------------------------
# FONCTIONS HELPER
# -------------------------
def create_session(table_id: str, mode: str, secret_word: str):
    """Créer une nouvelle session de jeu"""
    
    # Vérifier que le mot existe
    if secret_word not in embeddings:
        raise ValueError(f"Le mot '{secret_word}' n'existe pas dans le dictionnaire")
    
    # Calculer le ranking
    print(f"🔍 Calcul du ranking pour '{secret_word}' (table: {table_id})...")
    ranking = Jeu_complet.build_ranking(secret_word, embeddings)
    ranks = Jeu_complet.build_rank_dict(ranking)
    
    # Créer la session
    sessions[table_id] = {
        "secret_word": secret_word,
        "ranking": ranking,
        "ranks": ranks,
        "mode": mode,
        "attempts": 0,
        "guesses": []
    }
    
    print(f"✅ Session créée pour {table_id}")
    return sessions[table_id]

def get_session(table_id: str):
    """Récupérer une session existante"""
    if table_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Aucune partie en cours pour {table_id}")
    return sessions[table_id]

def delete_session(table_id: str):
    """Supprimer une session"""
    if table_id in sessions:
        del sessions[table_id]
        print(f"🗑️ Session supprimée: {table_id}")

# -------------------------
# ENDPOINTS
# -------------------------

@app.get("/")
def read_root():
    """Santé de l'API"""
    return {
        "status": "online",
        "words_loaded": len(embeddings),
        "themes_available": len(THEME_LIST),
        "themes": THEME_LIST,
        "active_sessions": len(sessions)
    }

@app.post("/new_game")
def new_game(request: NewGameRequest):
    """
    Démarrer une nouvelle partie
    
    Modes:
    - "theme_1" à "theme_19" : Thème spécifique
    - "theme_random" : Thème aléatoire
    - "random" : Mot aléatoire parmi les mots secrets
    - "hard" : Mot complètement aléatoire du dictionnaire
    """
    
    table_id = request.table_id
    mode = request.mode
    
    print(f"🎮 Nouvelle partie - Table: {table_id}, Mode: {mode}")
    
    # Supprimer l'ancienne session si elle existe
    if table_id in sessions:
        delete_session(table_id)
    
    secret_word = None
    
    try:
        # Mode Thème spécifique (theme_1, theme_2, ...)
        if mode.startswith("theme_") and mode != "theme_random":
            try:
                theme_index = int(mode.split("_")[1]) - 1
                
                if 0 <= theme_index < len(THEME_LIST):
                    selected_theme = THEME_LIST[theme_index]
                    secret_word = random.choice(themes[selected_theme])
                    print(f"   Thème: {selected_theme}")
                else:
                    raise HTTPException(status_code=400, detail="Index de thème invalide")
            except (ValueError, IndexError):
                raise HTTPException(status_code=400, detail="Format de thème invalide")
        
        # Mode Thème aléatoire
        elif mode == "theme_random":
            selected_theme = random.choice(THEME_LIST)
            secret_word = random.choice(themes[selected_theme])
            print(f"   Thème aléatoire: {selected_theme}")
        
        # Mode Random (parmi les mots secrets)
        elif mode == "random":
            all_words = []
            for words in themes.values():
                all_words.extend(words)
            secret_word = random.choice(all_words) if all_words else random.choice(list(embeddings.keys()))
        
        # Mode Hard (mot complètement aléatoire)
        elif mode == "hard":
            secret_word = random.choice(list(embeddings.keys()))
        
        else:
            raise HTTPException(status_code=400, detail="Mode inconnu")
        
        # Créer la session
        session = create_session(table_id, mode, secret_word)
        
        return {
            "success": True,
            "table_id": table_id,
            "mode": mode,
            "message": "Partie démarrée",
            "total_words": len(embeddings)
        }
    
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        print(f"❌ Erreur: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la création de la partie")

@app.post("/guess")
def guess_word(request: GuessRequest):
    """
    Deviner un mot
    
    Retourne:
    - rank: Position du mot (1 = mot secret)
    - temperature: Score de proximité
    - victory: True si c'est le mot secret
    - secret_word: Révélé uniquement si victoire
    """
    
    session = get_session(request.table_id)
    mot = request.mot.lower().strip()
    
    # Vérifier si le mot existe
    if mot not in embeddings:
        return {
            "error": "Mot inconnu",
            "found": False,
            "mot": mot
        }
    
    # Récupérer le rang
    rank = session["ranks"][mot]
    score = session["ranking"][rank - 1][1]
    temp = Jeu_complet.temperature(rank, len(session["ranking"]))
    
    # Incrémenter les tentatives
    session["attempts"] += 1
    session["guesses"].append({
        "mot": mot,
        "rank": rank,
        "temperature": temp
    })
    
    # Victoire ?
    victory = (rank == 1)
    
    response = {
        "success": True,
        "mot": mot,
        "rank": rank,
        "score": float(score),
        "temperature": float(temp),
        "found": True,
        "victory": victory,
        "attempts": session["attempts"]
    }
    
    if victory:
        response["secret_word"] = session["secret_word"]
        print(f"🎉 Victoire! Table {request.table_id} a trouvé '{session['secret_word']}'")
    
    return response

@app.post("/hint")
def get_hint_endpoint(request: HintRequest):
    """
    Obtenir un indice
    
    Niveaux:
    1 - Mot éloigné (rang 3000-5000)
    2 - Mot moyen (rang 500-1000)
    3 - Mot proche (rang 100-500)
    4 - Très proche (rang 2-20)
    """
    
    session = get_session(request.table_id)
    
    hint = Jeu_complet.get_hint(
        session["ranking"],
        request.level,
        session["ranks"]
    )
    
    if hint:
        session["attempts"] += 1  # Un indice compte comme une tentative
        return {
            "success": True,
            "hint": hint,
            "level": request.level
        }
    else:
        return {
            "success": False,
            "hint": "Aucun indice disponible à ce niveau",
            "level": request.level
        }

@app.get("/top/{table_id}/{count}")
def get_top(table_id: str, count: int = 10):
    """Obtenir le top N des mots les plus proches"""
    
    session = get_session(table_id)
    
    top_words = [
        {"mot": word, "score": float(score)}
        for word, score in session["ranking"][:count]
    ]
    
    return {
        "success": True,
        "top": top_words,
        "count": len(top_words)
    }

@app.post("/reveal/{table_id}")
def reveal_secret(table_id: str):
    """Révéler le mot secret (Give Up)"""
    
    session = get_session(table_id)
    secret = session["secret_word"]
    
    # Supprimer la session
    delete_session(table_id)
    
    return {
        "success": True,
        "secret_word": secret,
        "message": "Partie terminée - Abandon"
    }

@app.delete("/session/{table_id}")
def end_session(table_id: str):
    """Terminer une session"""
    
    delete_session(table_id)
    
    return {
        "success": True,
        "message": f"Session {table_id} terminée"
    }

@app.get("/check_word/{word}")
def check_word(word: str):
    """Vérifier si un mot existe dans le dictionnaire"""
    
    word = word.lower().strip()
    exists = word in embeddings
    
    return {
        "word": word,
        "exists": exists
    }

@app.get("/stats/{table_id}")
def get_stats(table_id: str):
    """Obtenir les statistiques d'une partie"""
    
    session = get_session(table_id)
    
    return {
        "success": True,
        "table_id": table_id,
        "mode": session["mode"],
        "attempts": session["attempts"],
        "guesses": session["guesses"][-10:]  # Les 10 derniers guess
    }

# -------------------------
# DÉMARRAGE
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

