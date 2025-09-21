from datasets import load_dataset
import pandas as pd
import random

# --- Step 1: Load the dataset subsets ---
print("Loading TREC-COVID dataset...")
queries = load_dataset("mteb/trec-covid", "queries")["queries"]
corpus = load_dataset("mteb/trec-covid", "corpus")["corpus"]
qrels = load_dataset("mteb/trec-covid", "default")["test"]

# --- Step 2: Convert to DataFrame ---
queries_df = pd.DataFrame(queries)
corpus_df = pd.DataFrame(corpus)
qrels_df = pd.DataFrame(qrels)

print("Qrels columns:", qrels_df.columns)

# --- Step 3: Sample 10 queries ---
sampled_queries = queries_df.sample(n=10, random_state=42)
sampled_query_ids = set(sampled_queries["_id"])

# --- Step 4: Get qrels for sampled queries ---
filtered_qrels = qrels_df[qrels_df["query-id"].isin(sampled_query_ids)]

# --- Step 5: Get 200 unique corpus docs linked to those queries ---
linked_docs = filtered_qrels["corpus-id"].unique().tolist()
if len(linked_docs) > 200:
    sampled_doc_ids = random.sample(linked_docs, 200)
else:
    sampled_doc_ids = linked_docs

filtered_corpus = corpus_df[corpus_df["_id"].isin(sampled_doc_ids)]

# --- Step 6: Filter qrels again for only selected docs ---
final_qrels = filtered_qrels[filtered_qrels["corpus-id"].isin(sampled_doc_ids)]

# --- Step 7: Save to CSV ---
sampled_queries.to_csv("queries.csv", index=False)
filtered_corpus.to_csv("corpus.csv", index=False)
final_qrels.to_csv("qrels.csv", index=False)

print("✅ Saved 10 queries -> queries.csv")
print("✅ Saved 200 corpus docs -> corpus.csv")
print("✅ Saved filtered qrels -> qrels.csv")
