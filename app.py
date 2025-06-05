from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import requests
import torch

app = Flask(__name__)

API_KEY = None
HEADERS = {"x-api-key": API_KEY} if API_KEY else {}

bi_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def search_papers(query, limit=10):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,abstract,url,citationCount,authors,year"
    }
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code == 200:
        return response.json().get("data", [])
    else:
        return []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results', methods=['GET', 'POST'])
def results():
    query = request.args.get('query') or request.form.get('query')
    if not query:
        return render_template('results.html', query="", results=[])

    papers = search_papers(query, limit=5)

    if not papers:
        return render_template('results.html', query=query, results=[])

    paper_texts = [(p.get('title') or '') + ". " + (p.get('abstract') or '') for p in papers]
    paper_embeddings = bi_encoder.encode(paper_texts, convert_to_tensor=True)
    query_embedding = bi_encoder.encode(query, convert_to_tensor=True)

    similarities = util.pytorch_cos_sim(query_embedding, paper_embeddings)[0]
    top_k = min(5, len(papers))
    top_results = torch.topk(similarities, k=top_k)

    cross_inp = [(query, paper_texts[i]) for i in top_results.indices]
    cross_scores = cross_encoder.predict(cross_inp)

    ranked = sorted(zip(top_results.indices.tolist(), cross_scores), key=lambda x: x[1], reverse=True)

    final_results = []
    for rank, (idx, score) in enumerate(ranked, start=1):
        paper = papers[idx]
        authors_list = [a['name'] for a in paper.get("authors", []) if "name" in a]
        if len(authors_list) > 3:
            authors = f"{authors_list[0]} et al."
        else:
            authors = ", ".join(authors_list)
        final_results.append({
            'rank': rank,
            'title': paper['title'],
            'url': paper['url'],
            'authors': authors,
            'score': round(float(score), 4),
            'citations': paper['citationCount'],
            'abstract': paper['abstract'],
            'year': paper['year']
        })

    return render_template('results.html', query=query, results=final_results) 

print("Flask app is starting...")
app.run(debug=True)
