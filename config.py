# --- Centralized Configuration for the Directory Analyzer ---

# --- Ollama and LLM Configuration ---
OLLAMA_MODEL = "tinyllama"
OLLAMA_REQUEST_TIMEOUT = 300.0

# --- Embedding Model Configuration ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Vector Store and Retriever Configuration ---
CHROMA_PERSIST_DIR = "./chroma_db_index"
TOP_K_CHUNKS = 3

# --- Indexing Configuration ---
ALLOWED_EXTENSIONS = [".log", ".txt", ".py", ".md"]
