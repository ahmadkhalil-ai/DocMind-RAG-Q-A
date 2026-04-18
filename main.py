from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from retriever import hybrid_retrieve, embedder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from models import AskRequest, AskResponse
from generator import generate_answer, stream_answer, ModelChoice
import faiss, numpy as np, shutil
import uvicorn, webbrowser, threading, time
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import tempfile, os

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

store = {}  # {filename: {"index": faiss_index, "texts": [str]}}


def load_and_chunk(file_path: str, chunk_size=512, overlap=64):
    ext = file_path.split(".")[-1].lower()
    loader = PyPDFLoader(file_path) if ext == "pdf" else TextLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " "]
    )
    return splitter.split_documents(docs)


def build_index(chunks):
    texts = [c.page_content for c in chunks]
    embeddings = embedder.encode(texts, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype("float32"))
    return index, texts


def hybrid_retrieve(query: str, faiss_index, texts: list, top_k=5, alpha=0.5):
    # Dense
    q_embed = embedder.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = faiss_index.search(q_embed, min(top_k * 2, len(texts)))
    dense_hits = {int(i): float(s) for i, s in zip(indices[0], scores[0]) if i != -1}

    # BM25
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(query.lower().split())
    max_bm25 = bm25_scores.max() + 1e-8

    # Fuse
    final = {}
    for idx in range(len(texts)):
        final[idx] = alpha * dense_hits.get(idx, 0) + (1 - alpha) * (bm25_scores[idx] / max_bm25)

    top = sorted(final, key=final.get, reverse=True)[:top_k]
    return [(texts[i], float(final[i])) for i in top]


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    tmp_dir = tempfile.gettempdir()  # works on Windows, Linux, Mac
    path = os.path.join(tmp_dir, file.filename)
    try:
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        chunks = load_and_chunk(path)
        index, texts = build_index(chunks)
        store[file.filename] = {"index": index, "texts": texts}
        return {"filename": file.filename, "chunks": len(chunks)}
    except Exception as e:
        return {"error": str(e)}


@app.post("/ask", response_model=AskResponse)
async def ask(payload: AskRequest):
    model = ModelChoice(payload.model or "claude")
    all_hits = []
    for doc in (payload.docs or list(store.keys())):
        if doc in store:
            hits = hybrid_retrieve(payload.query, store[doc]["index"], store[doc]["texts"])
            all_hits += [(doc, text, score) for text, score in hits]
    all_hits.sort(key=lambda x: x[2], reverse=True)
    return generate_answer(payload.query, all_hits[:5], model=model)


@app.post("/ask/stream")
async def ask_stream(payload: AskRequest):
    model = ModelChoice(payload.model or "claude")
    all_hits = []
    for doc in (payload.docs or list(store.keys())):
        if doc in store:
            hits = hybrid_retrieve(payload.query, store[doc]["index"], store[doc]["texts"])
            all_hits += [(doc, text, score) for text, score in hits]
    all_hits.sort(key=lambda x: x[2], reverse=True)
    return StreamingResponse(
        stream_answer(payload.query, all_hits[:5], model=model),
        media_type="text/plain"
    )


@app.get("/docs-list")
async def docs_list():
    return {"docs": [{"filename": k, "chunks": len(v["texts"])} for k, v in store.items()]}


@app.delete("/delete/{filename}")
async def delete_doc(filename: str):
    if filename in store:
        del store[filename]
        return {"deleted": filename}
    return {"error": "not found"}

@app.get("/ui")
def serve_ui():
    return FileResponse("index.html")


if __name__ == "__main__":

    def open_browser():
        time.sleep(1.5)  # wait for server to start
        webbrowser.open("http://localhost:8000/ui")
    
    threading.Thread(target=open_browser).start()
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)