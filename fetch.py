# === Semantic Scholar Citation Fetcher (with Year & Authors) ===
# Works in Colab, VS Code, or Anaconda
# - Uses 1 request/sec (per API key limit)
# - Fetches: citation count, URL, year, and authors
# - Includes cache, retry, and backoff
# - Processes first 200 titles (for testing)

import requests
import pandas as pd
import time
import json
import os
from tqdm import tqdm

# ---------------------------
# CONFIG
# ---------------------------
API_KEY = "O46KIJ2UH3650Mn5PLAhl1mD9UglVzuo49oRLZ78"  # 🔑 Your Semantic Scholar API key
API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "title,citationCount,year,url,authors.name"

INPUT_FILE = "corpus.csv"
OUTPUT_FILE = "corpus_complete.csv"
CACHE_FILE = "semantic_scholar_cache.json"

RATE_LIMIT_DELAY = 1.2     # 1 request/sec
MAX_RETRIES = 5
BACKOFF_FACTOR = 2.0
LIMIT_ROWS = 200           # ✅ limit for testing
REMOVE_OLD_CITATION_COL = True

# ---------------------------
# Load corpus
# ---------------------------
if not os.path.exists(INPUT_FILE):
    raise SystemExit(f"❌ Input file '{INPUT_FILE}' not found. Upload or place it here first.")

corpus_df = pd.read_csv(INPUT_FILE)
print(f"✅ Loaded {len(corpus_df)} rows from {INPUT_FILE}")

if "title" not in corpus_df.columns:
    raise SystemExit("❌ Input CSV must contain a 'title' column.")

# Limit dataset for safety
corpus_df = corpus_df.head(LIMIT_ROWS)
print(f"⚙️ Limiting to first {LIMIT_ROWS} titles (remove limit for full run).")

# Remove old citation-related columns
if REMOVE_OLD_CITATION_COL:
    drop_cols = [c for c in corpus_df.columns if any(k in c.lower() for k in ["citation", "url", "year", "author"])]
    if drop_cols:
        corpus_df = corpus_df.drop(columns=drop_cols, errors="ignore")
        print("🧹 Dropped old columns:", drop_cols)

# ---------------------------
# Load cache
# ---------------------------
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache = json.load(f)
    print(f"📦 Loaded {len(cache)} cached entries from {CACHE_FILE}")
else:
    cache = {}

# ---------------------------
# Helper: Fetch from API
# ---------------------------
def fetch_paper_data(title):
    title = str(title).strip()
    if not title:
        return (title, 0, "", "", "")

    # Use cache if available
    if title in cache:
        c = cache[title].get("citations", 0)
        url = cache[title].get("url", "")
        year = cache[title].get("year", "")
        authors = cache[title].get("authors", "")
        return (title, c, url, year, authors)

    headers = {"x-api-key": API_KEY}
    params = {"query": title, "fields": FIELDS, "limit": 1}

    retries = 0
    while retries < MAX_RETRIES:
        try:
            resp = requests.get(API_URL, headers=headers, params=params, timeout=20)
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                if data:
                    paper = data[0]
                    citations = int(paper.get("citationCount", 0))
                    url = paper.get("url", "")
                    year = paper.get("year", "")
                    authors_list = paper.get("authors", [])
                    authors = ", ".join(a["name"] for a in authors_list if "name" in a)
                else:
                    citations, url, year, authors = 0, "", "", ""
                cache[title] = {
                    "citations": citations,
                    "url": url,
                    "year": year,
                    "authors": authors
                }
                return (title, citations, url, year, authors)

            elif resp.status_code == 429:
                wait_time = BACKOFF_FACTOR ** retries
                print(f"⚠️ Rate limit hit. Waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                retries += 1
            else:
                retries += 1
                time.sleep(BACKOFF_FACTOR ** retries)

        except Exception as e:
            print(f"⚠️ Error fetching '{title}': {e}")
            retries += 1
            time.sleep(BACKOFF_FACTOR ** retries)

    # fallback
    cache[title] = {"citations": 0, "url": "", "year": "", "authors": ""}
    return (title, 0, "", "", "")

# ---------------------------
# Main loop
# ---------------------------
titles = corpus_df["title"].astype(str).tolist()
citations, urls, years, authors_list = [], [], [], []

print(f"🚀 Fetching citations, year, and authors for {len(titles)} titles (1 req/sec limit)...")

for title in tqdm(titles, desc="Fetching from Semantic Scholar"):
    title, c, u, y, a = fetch_paper_data(title)
    citations.append(c)
    urls.append(u)
    years.append(y)
    authors_list.append(a)
    time.sleep(RATE_LIMIT_DELAY)  # respect rate limit

# ---------------------------
# Save results
# ---------------------------
corpus_df["citations"] = citations
corpus_df["year"] = years
corpus_df["authors"] = authors_list
corpus_df["paper_url"] = urls

# Save cache and results
with open(CACHE_FILE, "w", encoding="utf-8") as f:
    json.dump(cache, f, ensure_ascii=False, indent=2)

corpus_df.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ Finished fetching metadata for {len(corpus_df)} titles")
print(f"💾 Saved updated corpus to {OUTPUT_FILE}")
print(f"📦 Cache stored in {CACHE_FILE}")

print("\n🔍 Sample output:")
print(corpus_df.head(5))
