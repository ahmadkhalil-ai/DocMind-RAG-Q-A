# DocMind

A RAG-powered document Q&A system. Upload a PDF, TXT, or Markdown file and ask questions about it in plain English. DocMind retrieves the most relevant passages using hybrid search and generates accurate answers with source citations.

## How it works

1. Upload a document — it gets chunked and indexed
2. Ask a question in natural language
3. The system retrieves the top matching chunks using FAISS (semantic) + BM25 (keyword) fusion
4. Claude or Llama generates an answer grounded in those chunks
5. The answer is returned with source citations and a confidence score

## Stack

- Python, FastAPI, LangChain
- FAISS + BM25 hybrid retrieval
- sentence-transformers (all-MiniLM-L6-v2) for embeddings
- Anthropic Claude Sonnet and Groq Llama 3.1 8B as generation models
- Vanilla HTML/CSS/JS frontend

## Setup

```
pip install fastapi uvicorn python-multipart langchain langchain-community langchain-anthropic langchain-text-splitters faiss-cpu sentence-transformers pypdf rank-bm25 pydantic python-dotenv groq aiofiles
```

Create a `.env` file:

```
ANTHROPIC_API_KEY=your_key
GROQ_API_KEY=your_key
```

## Run

```
python main.py
```

Opens at `http://localhost:8000/ui`

## Project structure

```
DocMind/
├── main.py          # FastAPI app, upload, ask endpoints
├── generator.py     # Claude and Groq generation logic
├── retriever.py     # Hybrid BM25 + FAISS retrieval
├── models.py        # Pydantic request/response schemas
├── index.html       # Frontend UI
├── .env             # API keys
└── requirements.txt
```