import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (two levels up from this file)
_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = "llama3-8b-8192"

CHROMA_PERSIST_DIR: str = str(_ROOT / "data" / "chroma")
CHROMA_COLLECTION: str = "echomaps_locations"

EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

WIKIPEDIA_API: str = "https://en.wikipedia.org/w/api.php"
WIKIDATA_SPARQL: str = "https://query.wikidata.org/sparql"

GEOSEARCH_RADIUS: int = 500   # metres
GEOSEARCH_LIMIT: int = 5

CHUNK_SIZE: int = 300
CHUNK_OVERLAP: int = 50

DEFAULT_N_RESULTS: int = 8
