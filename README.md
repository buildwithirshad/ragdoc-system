# RAG Document Retrieval System

A production-style Retrieval-Augmented Generation (RAG) system that lets you upload PDF documents and ask questions about them. Built with FastAPI, PostgreSQL (pgvector), Redis, and OpenAI.

## Architecture

```
PDF Upload → Chunking → OpenAI Embeddings → PostgreSQL (pgvector)
                              ↑
                         Redis Cache

User Query → Embed Query → Hybrid Search (Vector + FTS + RRF) → GPT Answer
```

## Features

- **PDF ingestion** — uploads parsed into token-aware chunks with page tracking
- **Hybrid search** — combines pgvector HNSW vector similarity and PostgreSQL full-text search, fused with Reciprocal Rank Fusion (RRF)
- **Redis embedding cache** — embeddings cached by content hash, eliminates redundant OpenAI API calls on re-ingestion
- **S3 storage** — original PDFs stored in AWS S3 with duplicate detection
- **Fully containerized** — Docker Compose spins up all services with a single command

## Tech Stack

- **API** — FastAPI, Uvicorn
- **Database** — PostgreSQL + pgvector (HNSW index)
- **Cache** — Redis
- **Storage** — AWS S3
- **AI** — OpenAI `text-embedding-3-small`, `gpt-4o-mini`
- **Infrastructure** — Docker, Docker Compose, AWS EC2

## Project Structure

```
rag-document-system/
├── app/
│   ├── main.py          # FastAPI app and routes
│   ├── config.py        # environment variables
│   ├── database.py      # PostgreSQL connection
│   ├── models.py        # SQLAlchemy table definitions
│   ├── chunker.py       # PDF parsing and token-aware chunking
│   ├── embedder.py      # OpenAI embeddings with Redis caching
│   ├── search.py        # hybrid search + GPT answer generation
│   └── s3.py            # S3 upload and duplicate detection
├── scripts/
│   └── init_db.sql      # pgvector extension + HNSW index
├── tests/
│   └── test_pipeline.py # 10 unit tests
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload` | Upload a PDF document |
| POST | `/ask` | Ask a question about uploaded documents |
| GET | `/documents` | List all uploaded documents |
| DELETE | `/documents/{id}` | Delete a document and its chunks |
| GET | `/health` | Health check |

## Getting Started

### Prerequisites
- Docker and Docker Compose
- OpenAI API key
- AWS account with S3 bucket and EC2 IAM role

### Run locally

```bash
git clone https://github.com/yourusername/rag-document-system
cd rag-document-system
cp .env.example .env
# Add your OpenAI API key to .env
docker compose up --build
```

Visit `http://localhost:8000/docs` for the interactive API docs.

## How It Works

### 1. Document Ingestion
- PDF is uploaded and saved temporarily to disk
- `pypdf` extracts text page by page, skipping blank pages
- `tiktoken` splits text into 512-token chunks with 64-token overlap so context is not lost at boundaries
- OpenAI generates embeddings for each chunk — cached in Redis by content hash to avoid redundant API calls
- Chunks and embeddings stored in PostgreSQL, original PDF uploaded to S3
- Duplicate detection — if the same filename already exists in S3, re-upload and re-processing are skipped entirely

### 2. Hybrid Search
- User query is embedded using the same OpenAI model (also cached in Redis)
- Two searches run against the chunks table:
  - **Vector search** — top 20 chunks by cosine similarity using the HNSW index on pgvector
  - **Full-text search** — top 20 chunks by PostgreSQL `ts_rank` using `plainto_tsquery`
- Results fused using **Reciprocal Rank Fusion (RRF)**: `score = 1/(60 + vector_rank) + 1/(60 + fts_rank)`
- Chunks appearing in both searches are ranked highest

### 3. Answer Generation
- Top K chunks passed to GPT as context with page numbers and filenames
- GPT answers using only the provided context
- `temperature=0` ensures deterministic, grounded answers

## Tests

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pytest tests/ -v
```

10 tests covering:
- Chunking logic (basic, overlap, empty input)
- Redis cache hit and miss behavior
- S3 upload and duplicate detection