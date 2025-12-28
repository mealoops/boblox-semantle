import numpy as np
import random
from collections import defaultdict
import gzip

# =========================
# PARAMETERS
# =========================
EMBEDDING_PATH = "glove_cemantle_filtered.txt.gz"
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
# LOAD GLOVE EMBEDDINGS (gzip)
# =========================
def load_glove(path, max_words):
    embeddings = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
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
# GAME MODE SELECTION
# =========================
def select_game_mode(themes):
    print("\n" + "=" * 60)
    print("🎮 CEMANTIX - Sélection du Mode de Jeu")
    print("=" * 60)
    print("\n1️⃣  Mode THÈME")
    print("2️⃣  Mode ALÉATOIRE")
    print("3️⃣  Mode DIFFICILE")
    print("0️⃣  Quitter")
    
    while True:
        choice = input("\n👉 Votre choix (1/2/3/0): ").strip()
        
        if choice == "0":
            print("👋 À bientôt!")
            return None, None
        
        elif choice == "1":
            return select_theme_mode(themes)
        
        elif choice == "2":
            all_words = []
            for words in themes.values():
                all_words.extend(words)
            
            if not all_words:
                print("❌ Aucun mot secret disponible!")
                return None, None
            
            secret = random.choice(all_words)
            print(f"\n✅ Mode ALÉATOIRE activé!\n")
            return "random", secret
        
        elif choice == "3":
            print(f"\n✅ Mode DIFFICILE activé!\n")
            return "hard", None
        
        else:
            print("❌ Choix invalide.")

def select_theme_mode(themes):
    print("\n" + "=" * 60)
    print("📚 SÉLECTION DU THÈME")
    print("=" * 60)
    
    theme_list = list(themes.keys())
    
    for i, theme in enumerate(theme_list, 1):
        word_count = len(themes[theme])
        print(f"{i:2}. {theme:20} ({word_count} mots)")
    
    print(" 0. Retour")
    
    while True:
        choice = input("\n👉 Choisir un thème: ").strip()
        
        if choice == "0":
            return select_game_mode(themes)
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(theme_list):
                selected_theme = theme_list[idx]
                secret = random.choice(themes[selected_theme])
                
                print(f"\n✅ Thème '{selected_theme}' sélectionné!\n")
                
                return selected_theme, secret
            else:
                print("❌ Numéro invalide.")
        except ValueError:
            print("❌ Entrez un numéro.")

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
    
    print(f"🔍 Calcul des similarités pour '{secret_word}'...")
    
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

# =========================
# MAIN GAME
# =========================
def main():
    print("🎮 CEMANTIX - Version Vectorielle Pure")
    print("=" * 60)
    print("Système de ranking basé sur:")
    print("  ✓ Similarité cosinus (80%)")
    print("  ✓ Distance euclidienne (20%)")
    print("=" * 60)
    
    themes = load_secret_words(SECRET_WORDS_PATH)
    if not themes:
        print("\n❌ Impossible de continuer sans fichier de mots secrets.")
        return
    
    print(f"\n⏳ Chargement des embeddings GloVe...")
    embeddings = load_glove(EMBEDDING_PATH, MAX_WORDS)
    print(f"✅ {len(embeddings)} mots chargés.")
    
    mode, secret_word = select_game_mode(themes)
    if mode is None:
        return
    if mode == "hard":
        secret_word = random.choice(list(embeddings.keys()))
    if secret_word not in embeddings:
        print(f"\n❌ Le mot '{secret_word}' n'existe pas dans le fichier GloVe!")
        return
    
    ranking = build_ranking(secret_word, embeddings)
    ranks = build_rank_dict(ranking)
    
    print("\n✅ Prêt!\n")
    print("💡 Commandes: mot | $hint1-4 | $top | $quit\n")
    
    attempts = 0
    
    while True:
        guess = input("🔎 Mot: ").lower().strip()
        
        if guess == "$quit":
            print(f"\n😔 Le mot était: {secret_word}")
            break
        
        if guess == "$top":
            print("\n🏆 Top 20:")
            for i, (word, score) in enumerate(ranking[:20], 1):
                print(f"   {i:2}. {word:15} (score: {score:.4f})")
            print()
            continue
        
        if guess.startswith("$hint"):
            try:
                level = int(guess[-1])
                hint = get_hint(ranking, level, ranks)
                attempts += 1
                if hint:
                    print(f"{hint}\n")
                else:
                    print("❌ Indice non disponible\n")
            except:
                print("❌ Utilisez $hint1, $hint2, $hint3 ou $hint4\n")
            continue
        
        attempts += 1
        
        if guess not in embeddings:
            print("❌ Mot inconnu\n")
            continue
        
        rank = ranks[guess]
        temp = temperature(rank, len(ranking))
        
        # Affichage minimaliste : mot, température, rang
        print(f"{guess:15} {temp:6.2f}°C   #{rank}\n")
        
        if rank == 1:
            print(f"🎉 GAGNÉ! '{secret_word}' en {attempts} essais!\n")
            break
    
    print("\n" + "=" * 60)
    print("📊 RÉSULTATS FINAUX")
    print("=" * 60)
    print(f"Essais: {attempts}")
    print("\n🔝 Top 30:")
    for i, (word, score) in enumerate(ranking[:30], 1):
        print(f"   {i:2}. {word:15} ({score:.4f})")

if __name__ == "__main__":
    main()
