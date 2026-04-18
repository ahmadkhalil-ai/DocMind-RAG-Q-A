from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from sentence_transformers import SentenceTransformer
import faiss, numpy as np, pickle, os

embedder = SentenceTransformer("all-MiniLM-L6-v2")  # free, fast, local

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
    index = faiss.IndexFlatIP(embeddings.shape[1])  # inner product = cosine on normalized
    index.add(embeddings.astype("float32"))
    return index, texts