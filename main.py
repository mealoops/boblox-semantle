from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os
import requests

from engine import GameEngine, GameManager

# =========================
# DOWNLOAD GLOVE IF NOT EXIST
# =========================
GLOVE_URL = "https://drive.usercontent.google.com/download?id=1iyn5o7PPsTneiP9Cg_fWQtYgG0R-wbE1&export=download&authuser=0&confirm=t&uuid=2c8bcf50-c931-4bda-87d5-6509b9856718&at=ANTm3cypfvRv4k_fNWYmKMWpcvaU:1768010086374"
GLOVE_PATH = "glove.6B.300d_no_numbers.txt"

if not os.path.exists(GLOVE_PATH):
    print("⏳ Téléchargement de GloVe...")
    r = requests.get(GLOVE_URL)
    with open(GLOVE_PATH, "wb") as f:
        f.write(r.content)
    print("✅ GloVe téléchargé")

# =========================
# LOAD DATA
# =========================
GameEngine.load_glove(GLOVE_PATH)
GameEngine.load_secrets("mots_secrets.txt")

manager = GameManager()
app = FastAPI(title="Boblox API")

# =========================
# REQUEST MODELS
# =========================
class CreateTableRequest(BaseModel):
    table_id: str

class PickRequest(BaseModel):
    table_id: str
    theme: Optional[str] = None  # pour pick_from_theme

class GuessRequest(BaseModel):
    table_id: str
    word: str

class HintRequest(BaseModel):
    table_id: str
    level: int

# =========================
# API ROUTES
# =========================
@app.post("/create")
def create_table(req: CreateTableRequest):
    if manager.get(req.table_id):
        raise HTTPException(status_code=400, detail="Table already exists")
    manager.create(req.table_id)
    return {"message": f"🎮 Table créée : {req.table_id}"}

@app.post("/cancel")
def cancel_table(req: CreateTableRequest):
    manager.cancel(req.table_id)
    return {"message": f"⛔ Table annulée : {req.table_id}"}

@app.post("/pick/glove")
def pick_glove(req: PickRequest):
    table = manager.get(req.table_id)
    if not table:
        raise HTTPException(status_code=404, detail="Table not found")
    table.pick_from_glove()
    return {"secret_word": table.secret_word}  # DEBUG: on peut cacher plus tard

@app.post("/pick/secrets")
def pick_secrets(req: PickRequest):
    table = manager.get(req.table_id)
    if not table:
        raise HTTPException(status_code=404, detail="Table not found")
    table.pick_from_secrets()
    return {"secret_word": table.secret_word}  # DEBUG

@app.post("/pick/theme")
def pick_theme(req: PickRequest):
    table = manager.get(req.table_id)
    if not table:
        raise HTTPException(status_code=404, detail="Table not found")
    if not req.theme or req.theme not in GameEngine.secret_words:
        raise HTTPException(status_code=400, detail="Theme not found")
    table.pick_from_theme(req.theme)
    return {"secret_word": table.secret_word}  # DEBUG

@app.post("/guess")
def guess_word(req: GuessRequest):
    table = manager.get(req.table_id)
    if not table:
        raise HTTPException(status_code=404, detail="Table not found")
    return table.guess(req.word)

@app.post("/hint")
def get_hint(req: HintRequest):
    table = manager.get(req.table_id)
    if not table:
        raise HTTPException(status_code=404, detail="Table not found")
    hint = table.hint(req.level)
    return {"hint": hint}

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


