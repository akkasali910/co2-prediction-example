import os
import sys
from pathlib import Path
import torch
import shutil

# --- LlamaIndex Imports ---
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# --- Disable Telemetry/Warnings ---
os.environ["LLAMA_INDEX_DO_NOT_TRACK"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Configuration ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ALLOWED_EXTENSIONS = [".log", ".txt", ".py", ".md"]
CHROMA_PERSIST_DIR = "./chroma_db_index" # Directory to save the persistent index
# ---------------------

def build_persistent_index():
    """
    Reads files from the local directory, creates vector embeddings,
    and stores them in a persistent ChromaDB index on disk.
    """
    # Determine the device for PyTorch (used by HuggingFace Embeddings)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n--- 1. Initializing Embeddings on Device: {device.upper()} ---")

    try:
        # Initialize Embeddings (HuggingFace)
        embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME, device=device)
        Settings.embed_model = embed_model
        print("✅ Embeddings initialized successfully.")
    except Exception as e:
        print(f"❌ Error initializing HuggingFace Embeddings. Check PyTorch/CUDA installation. Error: {e}")
        sys.exit(1)

    print("\n--- 2. Reading Directory Files ---")
    try:
        reader = SimpleDirectoryReader(input_dir=".", required_exts=ALLOWED_EXTENSIONS, recursive=True)
        documents = reader.load_data()

        if not documents:
            print(f"⚠️ No files found in the current directory with extensions: {', '.join(ALLOWED_EXTENSIONS)}.")
            sys.exit(0)

        indexed_files = [Path(doc.metadata['file_path']).name for doc in documents]
        print(f"✅ Found and loaded {len(indexed_files)} files: {', '.join(indexed_files)}")

    except Exception as e:
        print(f"❌ Error reading directory files: {e}")
        sys.exit(1)

    print(f"\n--- 3. Building Persistent Vector Index at '{CHROMA_PERSIST_DIR}' ---")
    # Clear out the old directory if it exists
    if os.path.exists(CHROMA_PERSIST_DIR):
        shutil.rmtree(CHROMA_PERSIST_DIR)

    # Create a persistent ChromaDB client
    db = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    chroma_collection = db.get_or_create_collection("directory_analysis_collection")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Build and persist the index
    VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    print("✅ Index built and saved successfully.")
    print("\n=======================================================")
    print("      Indexing Complete. You can now run the analyzer or MCP server.")
    print("=======================================================\n")

if __name__ == '__main__':
    build_persistent_index()