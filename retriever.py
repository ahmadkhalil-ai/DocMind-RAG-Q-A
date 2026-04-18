from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2") 

def hybrid_retrieve(query, faiss_index, texts, top_k=5, alpha=0.5):
    # Dense retrieval
    q_embed = embedder.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = faiss_index.search(q_embed, top_k * 2)
    dense_hits = {i: s for i, s in zip(indices[0], scores[0])}

    # BM25 sparse retrieval
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(query.lower().split())

    # RRF fusion
    final = {}
    for idx in range(len(texts)):
        rrf = alpha * dense_hits.get(idx, 0) + (1 - alpha) * (bm25_scores[idx] / (bm25_scores.max() + 1e-8))
        final[idx] = rrf

    top = sorted(final, key=final.get, reverse=True)[:top_k]
    return [(texts[i], float(final[i])) for i in top]