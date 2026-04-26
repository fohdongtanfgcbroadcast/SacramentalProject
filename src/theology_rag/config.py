from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"
METADATA_DIR = DATA_ROOT / "metadata"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

EMBEDDING_MODEL = "BAAI/bge-m3"
CLAUDE_MODEL = "claude-opus-4-7"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
