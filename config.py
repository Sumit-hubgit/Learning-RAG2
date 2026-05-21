from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    REDIS_HOST = "127.0.0.1"
    REDIS_PORT = 6379
    REDIS_URL = "redis://127.0.0.1:6379"
    INDEX_NAME = "advanced-rag-index"
    INDEX_FLAG = "advanced_rag_ready"
    CHUNK_FILE = "chunks.pkl"
    PDF_DIR = "data/pdf"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    LLM_MODEL = "llama-3.1-8b-instant"
    TOP_K_RETRIEVE = 10
    TOP_K_RERANK = 5
    CACHE_TTL = 3600