from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from rank_bm25 import BM25Okapi
from collections import defaultdict
import pandas as pd
import torch
import math

app = Flask(__name__)

# --- Load CSV Files with Citation Info ---
print("Loading corpus, queries, and qrels...")

corpus_df = pd.read_csv("corpus.csv")
queries_df = pd.read_csv("queries.csv")
qrels_df = pd.read_csv("qrels.csv")

corpus = {
    row["doc_id"]: {
        "title": row["title"],
        "text": row["text"],
        "citations": int(row.get("citations", 0))
    }
    for _, row in corpus_df.iterrows()
}

queries = {
    row["query_id"]: row["text"]
    for _, row in queries_df.iterrows()
}

qrels = defaultdict(dict)
for _, row in qrels_df.iterrows():
    qrels[str(row["query_id"])][row["doc_id"]] = int(row["score"])

# --- Load Models ---
print("Loading models...")
bi_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# --- Build BM25 Index ---
print("Building BM25 index...")
tokenized_corpus = []
doc_id_list = []

for doc_id, doc in corpus.items():
    combined_text = (doc['title'] or '') + ". " + (doc['text'] or '')
    tokens = combined_text.lower().split()
    tokenized_corpus.append(tokens)
    doc_id_list.append(doc_id)

bm25 = BM25Okapi(tokenized_corpus)

# --- Evaluation Metrics ---
def compute_dcg(relevances):
    return sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(relevances))

def compute_ndcg(predicted_rels, ideal_rels):
    dcg = compute_dcg(predicted_rels)
    idcg = compute_dcg(sorted(ideal_rels, reverse=True))
    return dcg / idcg if idcg != 0 else 0

def compute_mrr(relevances):
    for i, rel in enumerate(relevances):
        if rel > 0:
            return 1 / (i + 1)
    return 0

# --- Citation-aware nFAIRR ---
def compute_nfairr_fair(ranked_doc_ids, query_id):
    rel_map = qrels.get(query_id, {})
    citation_scores = []

    for rank, doc_id in enumerate(ranked_doc_ids):
        rel = rel_map.get(doc_id, 0)
        if rel > 0:
            citation = corpus[doc_id].get("citations", 1)
            fairness_weight = 1 / math.log1p(citation)
            score = fairness_weight / (rank + 1)
            citation_scores.append(score)

    # Ideal fairness if low-cited relevant docs came first
    ideal_order = sorted([
        (1 / math.log1p(corpus[doc_id].get("citations", 1)), doc_id)
        for doc_id in rel_map.keys() if rel_map[doc_id] > 0
    ], reverse=True)

    ideal_scores = [w / (i + 1) for i, (w, _) in enumerate(ideal_order)]

    actual = sum(citation_scores)
    ideal = sum(ideal_scores)
    return actual / ideal if ideal != 0 else 0

# --- Citation-aware BM25 + Cross-Encoder Search ---
def search_local(query_text, top_k=5, bm25_k=20, bi_k=10):
    query_tokens = query_text.lower().split()
    raw_bm25_scores = bm25.get_scores(query_tokens)

    # --- Stage 1: BM25 Top-K (pure BM25, no citation weighting)
    bm25_top_indices = sorted(
        range(len(raw_bm25_scores)),
        key=lambda i: raw_bm25_scores[i],
        reverse=True
    )[:bm25_k]

    bm25_top_doc_ids = [doc_id_list[i] for i in bm25_top_indices]
    bm25_top_texts = [
        (corpus[doc_id]["title"] or '') + ". " + (corpus[doc_id]["text"] or '')
        for doc_id in bm25_top_doc_ids
    ]

    # --- Stage 2: Bi-Encoder Semantic Re-ranking
    query_embedding = bi_encoder.encode(query_text, convert_to_tensor=True)
    doc_embeddings = bi_encoder.encode(bm25_top_texts, convert_to_tensor=True)
    bi_scores = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]

    bi_top_indices = torch.topk(bi_scores, k=min(bi_k, len(bi_scores))).indices.tolist()
    bi_top_doc_ids = [bm25_top_doc_ids[i] for i in bi_top_indices]
    bi_top_texts = [bm25_top_texts[i] for i in bi_top_indices]

    # --- Stage 3: Cross-Encoder Final Ranking
    cross_inputs = [(query_text, doc_text) for doc_text in bi_top_texts]
    cross_scores = cross_encoder.predict(cross_inputs)

    ranked = sorted(
        zip(bi_top_doc_ids, cross_scores),
        key=lambda x: x[1],
        reverse=True
    )
    return ranked


# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html', queries=queries)

@app.route('/results', methods=['GET', 'POST'])
def results():
    selected_query_id = request.args.get('query_id') or request.form.get('query_id')
    if not selected_query_id:
        return render_template('results.html', query="", results=[], queries=queries)

    query_text = queries[selected_query_id]
    ranked = search_local(query_text, top_k=5)

    rel_map = qrels.get(selected_query_id, {})
    predicted_rels = []
    final_results = []
    ranked_doc_ids = []

    for rank, (doc_id, score) in enumerate(ranked, start=1):
        doc = corpus[doc_id]
        rel_score = rel_map.get(doc_id, 0)
        predicted_rels.append(rel_score)
        ranked_doc_ids.append(doc_id)

        final_results.append({
            'rank': rank,
            'title': doc['title'],
            'abstract': doc['text'],
            'url': f"https://www.semanticscholar.org/paper/{doc_id}",
            'score': round(float(score), 4),
            'doc_id': doc_id,
            'citations': doc.get("citations", 0)
        })

    ideal_rels = sorted(rel_map.values(), reverse=True)
    ndcg = compute_ndcg(predicted_rels, ideal_rels)
    mrr = compute_mrr(predicted_rels)
    nfairr = compute_nfairr_fair(ranked_doc_ids, selected_query_id)

    return render_template('results.html', query=query_text, results=final_results,
                           ndcg=round(ndcg, 4), mrr=round(mrr, 4), nfairr=round(nfairr, 4),
                           queries=queries)

# --- Run App ---
if __name__ == '__main__':
    print("Flask app is starting...")
    app.run(debug=True)
