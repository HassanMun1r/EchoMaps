# EchoMaps

A geospatial multi-agent RAG system that generates layered historical and cultural
narratives for any location on a map. 100% free to run — no paid APIs or services.

## Stack

| Layer | Technology |
|---|---|
| Frontend | Leaflet.js + Vanilla JS (single HTML file) |
| Backend | FastAPI (Python) |
| LLM | Groq API — llama3-8b-8192 (free tier) |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 (local) |
| Vector DB | ChromaDB (local, persistent) |
| Data | Wikipedia REST API + Wikidata SPARQL (free, no key) |
| Map tiles | OpenStreetMap via Leaflet |

## Setup

### 1. Clone and create virtual environment

```bash
git clone <repo>
cd echomaps
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and add your Groq API key (free at https://console.groq.com)
```

### 4. Start the backend

```bash
uvicorn backend.main:app --reload
```

### 5. Open the frontend

Open `frontend/index.html` in your browser (no build step needed).

Click anywhere on the map, then click **Discover** to fetch historical narratives.

## Project Structure

```
echomaps/
  backend/
    main.py              # FastAPI app entry point
    config.py            # env vars and constants
    rag/
      fetcher.py         # Wikipedia + Wikidata data fetching
      embedder.py        # sentence-transformers embedding
      vectorstore.py     # ChromaDB setup and operations
      retriever.py       # semantic search and chunk retrieval
    api/
      routes.py          # FastAPI route definitions
  frontend/
    index.html           # full map UI (single file)
  data/
    chroma/              # local vector store (auto-created)
  .env.example
  requirements.txt
```

## API Endpoints

- `POST /api/location` — `{lat, lon, query}` → `{context, sources, location_id}`
- `GET /health` — `{status: "ok"}`

## Getting a Groq API Key

1. Go to https://console.groq.com
2. Sign up (free)
3. Create an API key
4. Paste it into `.env` as `GROQ_API_KEY=...`
