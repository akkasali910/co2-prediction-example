import os
import sys
from pathlib import Path
import torch

# --- LlamaIndex Imports ---
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
# IMPORTANT: Use chromadb.Client for in-memory, non-persistent storage
# This avoids "read-only database" errors in restricted environments.
import chromadb

# --- Configuration (Max Speed Priority) ---
OLLAMA_MODEL = "tinyllama" 
OLLAMA_REQUEST_TIMEOUT = 300.0 
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ALLOWED_EXTENSIONS = [".log", ".txt", ".py", ".md"]
TOP_K_CHUNKS = 1
# ------------------------------------------

def setup_rag_engine():
    """Initializes and configures the LlamaIndex RAG pipeline, prioritizing GPU usage and memory stability."""
    
    # Determine the device for PyTorch (used by HuggingFace Embeddings)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n--- 1. Initializing Embeddings on Device: {device.upper()} ---")
    
    try:
        # 1. Initialize Embeddings (HuggingFace) and force it to use CUDA if available
        embed_model = HuggingFaceEmbedding(
            model_name=EMBEDDING_MODEL_NAME,
            device=device
        )
        print("✅ Embeddings initialized successfully.")
    except Exception as e:
        print(f"❌ Error initializing HuggingFace Embeddings. Check PyTorch/CUDA installation. Error: {e}")
        sys.exit(1)

    print("\n--- 2. Initializing Ollama (tinyllama) ---")
    try:
        # 2. Initialize LLM (Ollama) - Ollama handles its own GPU configuration
        llm = Ollama(model=OLLAMA_MODEL, request_timeout=OLLAMA_REQUEST_TIMEOUT) 
        llm.complete("Hi", max_tokens=1) # Test call to verify model access
        
        Settings.llm = llm
        Settings.embed_model = embed_model
        
        print(f"✅ Ollama LLM '{OLLAMA_MODEL}' initialized successfully.")
    except Exception as e:
        print(f"❌ Ollama ERROR: Model '{OLLAMA_MODEL}' not accessible or server not running.")
        print(f"   Action: Ensure 'ollama serve' is running and you have pulled the model using: 'ollama pull {OLLAMA_MODEL}'")
        sys.exit(1)
        
    print("\n--- 3. Reading Directory Files ---")
    # 3. Document Loading
    try:
        # The reader is still relative to the current working directory
        reader = SimpleDirectoryReader(
            input_dir=".", 
            required_exts=ALLOWED_EXTENSIONS,
            recursive=True
        )
        documents = reader.load_data()
        
        if not documents:
            print(f"⚠️ No files found in the current directory with extensions: {", ".join(ALLOWED_EXTENSIONS)}.")
            sys.exit(0)
            
        indexed_files = [Path(doc.metadata['file_path']).name for doc in documents]
        print(f"✅ Found and loaded {len(indexed_files)} files: {', '.join(indexed_files)}")
            
    except Exception as e:
        print(f"❌ Error reading directory files: {e}")
        sys.exit(1)

    print("\n--- 4. Building Vector Index (IN-MEMORY ChromaDB) ---")
    # 4. Vector Store Setup (IN-MEMORY ChromaDB)
    # Using chromadb.Client() for in-memory, avoiding file permission errors.
    db_client = chromadb.Client()
    chroma_collection = db_client.get_or_create_collection("directory_analysis_collection")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 5. Build Index
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    print("✅ Index built in memory.")

    # 6. Setup Query Engine
    system_prompt = (
        "You are an expert file analyzer. Your task is to analyze the provided file "
        "snippets and answer the user's question concisely. The file snippets are prefixed "
        "with their original filename for context. "
        "Only use the information from the file content provided in the context. "
        "If the answer is not in the context, state that you cannot find the answer in the provided files."
    )
    
    query_engine = index.as_query_engine(
        system_prompt=system_prompt,
        similarity_top_k=TOP_K_CHUNKS, 
        streaming=True
    )
    
    return query_engine

def analyze_console():
    """Main console loop for querying the RAG engine."""
    
    query_engine = setup_rag_engine()
    
    print("\n=======================================================")
    print("      Directory Analyzer is READY (MAX SPEED)")
    print("=======================================================")
    print("   Enter your query or type 'exit' or 'quit' to end.")
    print("=======================================================\n")
    
    while True:
        try:
            user_query = input("❓ Your Query: ")
            
            if user_query.lower() in ['exit', 'quit']:
                print("👋 Exiting analyzer. Goodbye!")
                break
            
            if not user_query.strip():
                continue

            print("\n🤖 AI Response (Streaming):")
            
            # Use query() which returns a streamable response object
            response_stream = query_engine.query(user_query)

            # Iterate over the tokens in the response stream
            for token in response_stream.response_gen:
                print(token, end="", flush=True)

            print("\n-------------------------------------------------------\n")
            
        except KeyboardInterrupt:
            print("\n👋 Exiting analyzer. Goodbye!")
            break
        except Exception as e:
            # We catch specific PyTorch errors related to memory/device to give better feedback
            if "CUDA out of memory" in str(e):
                 print("\n❌ Error: CUDA out of memory. Try closing other GPU applications or reducing context size.")
            else:
                print(f"\n❌ Error during query execution: {e}")
                print("   Please ensure your Ollama server is running and accessible.")

if __name__ == '__main__':
    analyze_console()

