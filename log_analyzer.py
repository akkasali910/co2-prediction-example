import os
import sys

# --- LlamaIndex Imports ---
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# --- Local Imports ---
from llama_index.core.chat_engine import ContextChatEngine
from config import (OLLAMA_MODEL, OLLAMA_REQUEST_TIMEOUT, EMBEDDING_MODEL_NAME, 
                    CHROMA_PERSIST_DIR, TOP_K_CHUNKS)

# --- Disable Telemetry/Warnings ---
os.environ["LLAMA_INDEX_DO_NOT_TRACK"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def setup_chat_engine():
    """Initializes the LlamaIndex chat engine."""
    print("\n--- 1. Initializing Ollama LLM ---")
    try:
        llm = Ollama(
            model=OLLAMA_MODEL,
            request_timeout=OLLAMA_REQUEST_TIMEOUT,
            max_tokens=512,
            stop_sequences=["\nUser:", "\n\n"],
            temperature=0.1
        )
        llm.complete("Hi", max_tokens=1) # Test call
        print(f"✅ Ollama LLM '{OLLAMA_MODEL}' initialized successfully.")
    except Exception as e:
        print(f"❌ Ollama ERROR: Model '{OLLAMA_MODEL}' not accessible or server not running.")
        print(f"   Action: Ensure 'ollama serve' is running and you have pulled the model using: 'ollama pull {OLLAMA_MODEL}'")
        sys.exit(1)

    print("\n--- 2. Initializing Retriever from Local Index ---")
    try:
        # Initialize Embeddings
        Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
        Settings.llm = llm

        # Load the persistent ChromaDB index
        db = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        chroma_collection = db.get_or_create_collection("directory_analysis_collection")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

        # Create a retriever from the index
        retriever = index.as_retriever(similarity_top_k=TOP_K_CHUNKS)
        print("✅ Retriever initialized successfully.")
    except Exception as e:
        print(f"❌ Failed to initialize retriever from '{CHROMA_PERSIST_DIR}'.")
        print(f"   Action: Ensure you have run 'python build_index.py' first. Error: {e}")
        sys.exit(1)

    print("\n--- 3. Initializing LlamaIndex Chat Engine ---")
    # Define a custom system prompt for the chat engine
    system_prompt = (
        "You are an expert file analyzer. Your task is to answer the user's questions based on the provided context. "
        "The context is retrieved from a local directory of files. "
        "Always be helpful and answer concisely. If the answer is not in the context, "
        "state that you cannot find the answer in the provided files. "
        "You have access to the conversation history."
    )
    
    chat_engine = ContextChatEngine.from_defaults(
        retriever=retriever,
        llm=llm,
        system_prompt=system_prompt,
    )
    print("✅ Chat Engine is ready.")
    return chat_engine

def analyze_console():
    """Main console loop for chatting about the local index."""
    chat_engine = setup_chat_engine()
    try:
        print("\n=======================================================")
        print("      Directory Analyzer is READY (Conversational Mode)")
        print("=======================================================")
        print("   - Using local LlamaIndex and ChromaDB index.")
        print("   - Enter your query or type 'exit' or 'quit' to end. Type 'reset' to clear history.")
        print("=======================================================\n")
        
        while True:
            user_query = input("❓ Your Query: ")
            
            if user_query.strip().lower() in ['exit', 'quit']:
                break
            
            if not user_query.strip():
                continue

            print("\n🤖 AI Response (Streaming):")
            
            if user_query.strip().lower() == 'reset':
                chat_engine.reset()
                print("Conversation history has been reset.")
                print("-------------------------------------------------------\n")
                continue

            # Use the stream_chat() method which returns a StreamingResponse object
            streaming_response = chat_engine.stream_chat(user_query)

            for token in streaming_response.response_gen:
                print(token, end="", flush=True)

            print("\n-------------------------------------------------------\n")
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
    finally:
        print("\n👋 Exiting analyzer. Goodbye!")

if __name__ == '__main__':
    analyze_console()
