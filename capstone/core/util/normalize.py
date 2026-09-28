import asyncio
import aiohttp
import time
import os
import spacy
import re

# --- CONFIGURATION ---
USER_AGENT = "EduCapstoneBot/1.0 (dev_mode@student.edu)"
API_URL = "https://www.wikidata.org/w/api.php"
LIMIT = 10

# Resolve subjects.csv path relative to this file
_self_dir = os.path.dirname(os.path.abspath(__file__))
SUBJECT_FILE = os.path.join(_self_dir, "subjects.csv")

# --- 1. SETUP SPACY & STOPWORDS ---
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("⚠️ Downloading Spacy model...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Các tính từ chung chung cần loại bỏ
GENERIC_ADJECTIVES = {
    "good", "bad", "great", "poor", "excellent",
    "simple", "complex", "basic", "advanced", 
    "general", "specific", "various", "different",
    "main", "key", "important", "significant", "the", "a", "an"
}

# --- 2. UTILS ---
if not os.path.exists(SUBJECT_FILE):
    os.makedirs(os.path.dirname(SUBJECT_FILE), exist_ok=True)
    with open(SUBJECT_FILE, "w", encoding="utf-8") as f:
        defaults = [
            "programming,1.0", "software,1.0", "algorithm,1.0", "computer,1.0", "science,1.0", 
            "mathematics,1.0", "university,1.0", "technology,1.0", "framework,1.0", "library,1.0",
            "education,1.0", "intelligence,1.0", "learning,1.0", "data,1.0", "system,0.5", "city,0.5",
            "field,0.5", "discipline,0.5", "branch,0.5", "concept,0.5", "method,0.5", "term,0.5"
        ]
        f.write("\n".join(defaults))

def load_filter_scores(filepath):
    scores = {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "," in line:
                    parts = line.split(",", 1)
                    word = parts[0].strip().lower()
                    if word in ("word", "keyword", "subject", "term"):
                        continue
                    try:
                        score = float(parts[1].strip())
                    except ValueError:
                        score = 1.0
                else:
                    word = line.lower()
                    score = 1.0
                
                if word:
                    scores[word] = score
    except Exception:
        pass
    return scores
def invalid_word(text: str) -> bool:
        if re.search(r'\d{3,}', text): return True
        if len(text) < 2: return True
        return False

def nlp_normalize(text: str, doc = None) -> str:
    """
    Cleaning concepts for subsequence insertion and Wikidata search, avoiding duplication of different writing withy the same concept.
    """
    if invalid_word(text):
        return ""
    doc = doc or nlp(text)
    clean_tokens = []
    
    for token in doc:
        if token.pos_ in ["PUNCT", "SPACE"]:
            continue
            
        # Bỏ mạo từ (DET) đầu câu hoặc tính từ chung chung (ADJ)
        if (token.pos_ == "DET") or \
           (token.pos_ == "ADJ" and token.lemma_.lower() in GENERIC_ADJECTIVES):
            continue

        # Lemmatize: Đưa danh từ số nhiều về số ít
        if token.tag_ == "NNS":
            clean_tokens.append(token.lemma_)
        else:
            clean_tokens.append(token.text)
            
    result = " ".join(clean_tokens)

    return result.title() if result.strip() else text.title()

# --- 3. ASYNC WORKER ---
async def _fetch_single(session, semaphore, search_text, filter_scores):

    params = {
        "action": "wbsearchentities",
        "format": "json",
        "language": "en",
        "search": search_text,
        "limit": 20,
        "type": "item"
    }
    article_kws = ["article", "journal", "publication", "thesis", "paper"]
    async with semaphore:
        try:
            async with session.get(API_URL, params=params) as response:
                if response.status != 200:
                    return search_text, {"id": None, "error": f"HTTP {response.status}"}
                
                data = await response.json()
                candidates = data.get("search", [])
                
                if not candidates:
                    return search_text, {"id": None, "error": "Not found", "used_query": search_text}
                
                best_cand = None
                best_score = 0.0
                article_desc = ""
                
                for i, cand in enumerate(candidates):
                    desc = cand.get("description", "").lower()
                    
                    current_score = 0.0
                    for kw, score in filter_scores.items():
                        if re.search(r'\b' + re.escape(kw) + r'\b', desc):
                            current_score += score
                            
                    if i < 10 and not article_desc:
                        if any(re.search(r'\b' + re.escape(ak) + r'\b', desc) for ak in article_kws):
                            article_desc = cand.get("description", "")
                            
                    if current_score > best_score:
                        best_score = current_score
                        best_cand = cand
                
                if best_cand and best_score >= 1.0:
                    result_desc = best_cand.get("description", "")
                    if article_desc:
                        result_desc += f"\n**ARTICLE: {article_desc}"
                    
                    return search_text, {
                        "id": best_cand["id"],
                        "label": best_cand["label"],
                        "desc": result_desc,
                        "status": "MATCH"
                    }
                
                return search_text, {"id": None, "error": "No semantic match with score >= 1.0"}
                    
        except Exception as e:
            return search_text, {"id": None, "error": str(e)}

async def wiki_resolver(queries: list, filter_filepath: str =SUBJECT_FILE , concurrency: int = None) -> dict:
    filter_scores = load_filter_scores(filter_filepath)
    if concurrency is None:
        concurrency = LIMIT
    semaphore = asyncio.Semaphore(concurrency)
    
    headers = {"User-Agent": USER_AGENT}
    
    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = []
        for q in queries:

            tasks.append(_fetch_single(session, semaphore, q, filter_scores))
            
        results = await asyncio.gather(*tasks)
        return {k: v for k, v in results}

# --- 4. TEST ---
if __name__ == "__main__":
    real_queries = [
        "The Good Algorithms",       # -> Algorithm
        "Advanced Python Programming", # -> Python Programming
        "Simple Linear Regression",  # -> Linear Regression
        "Decision Trees",            # -> Decision Tree (Số ít)
        "Hanoi",
        "Mars",
        "Deep Learning"              # -> Deep Learning (Giữ nguyên)
    ]

    print(f"Processing {len(real_queries)} queries...")
    start = time.time()
    
    # Windows fix cho asyncio loop
    try:
        results = asyncio.run(wiki_resolver(real_queries, SUBJECT_FILE))
    except RuntimeError: # Jupyter/Special enviroment handling
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(wiki_resolver(real_queries, SUBJECT_FILE))

    end = time.time()
    print(f"Completed in: {end - start:.2f} seconds\n")
    
    for query in real_queries:
        data = results.get(query)
        print(f"Origin: '{query}'")
        if data:
            print(f" -> Cleaned: '{data.get('cleaned_name')}'")
            print(f" -> Found: {data.get('label')} ({data.get('id')}) - {data.get('desc')}")
        else:
            print(" -> Error/Not found")
        print("-" * 40)