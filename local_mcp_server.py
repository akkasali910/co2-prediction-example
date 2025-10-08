import os
import logging
from flask import Flask, request, jsonify

import argparse
# --- LlamaIndex Imports ---
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# --- Disable Telemetry/Warnings ---
os.environ["LLAMA_INDEX_DO_NOT_TRACK"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Configuration ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_PERSIST_DIR = "./chroma_db_index"
TOP_K_CHUNKS = 3 # How many context chunks to retrieve
# ---------------------

# --- Flask App Setup ---
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# --- Global variable for the retriever ---
# This will be initialized once on the first request
retriever = None

def initialize_retriever():
    """
    Loads the persistent vector index and initializes a retriever.
    This function is designed to run only once.
    """
    global retriever
    if retriever is None:
        logging.info("Initializing retriever for the first time...")
        try:
            # 1. Initialize Embeddings
            Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)

            # 2. Load the persistent ChromaDB index
            db = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
            chroma_collection = db.get_or_create_collection("directory_analysis_collection")
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

            # 3. Create a retriever from the index
            retriever = index.as_retriever(similarity_top_k=TOP_K_CHUNKS)
            logging.info("✅ Retriever initialized successfully.")
        except Exception as e:
            logging.error(f"❌ Failed to initialize retriever. Did you run 'build_index.py' first? Error: {e}")
            raise

@app.route('/mcp/v1/get_model_context', methods=['POST'])
def get_model_context():
    """
    MCP-compliant endpoint to retrieve context from the vector store.
    Can handle a single query ("user_text") or multiple queries ("user_texts").
    """
    try:
        initialize_retriever() # Ensure retriever is ready
    except Exception as e:
        return jsonify({"error": f"Could not initialize knowledge source: {e}"}), 500

    data = request.get_json()
    # Support both a single query and a list of queries
    user_text = data.get("user_text")
    user_texts = data.get("user_texts")

    queries = []
    if user_texts and isinstance(user_texts, list):
        queries.extend(user_texts)
    elif user_text:
        queries.append(user_text)

    if not queries:
        return jsonify({"error": "A 'user_text' string or a 'user_texts' list is required"}), 400

    logging.info(f"Received {len(queries)} queries: {queries}")

    # Retrieve nodes for all queries and de-duplicate them
    all_nodes = {}
    for query in queries:
        retrieved_nodes = retriever.retrieve(query)
        for node in retrieved_nodes:
            # Use node_id as a key to ensure uniqueness
            if node.node_id not in all_nodes:
                all_nodes[node.node_id] = node

    # Format the response according to the Model Context Protocol
    context_items = [
        {"item_type": "text", "content": node.get_content()}
        for node in all_nodes.values()
    ]

    response = {"context_items": context_items}
    return jsonify(response)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a local MCP server.")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the server on.")
    args = parser.parse_args()

    print("🚀 Starting Local MCP Server...")
    print("   - Make sure you have run 'pip install Flask'.")
    print(f"   - Ensure the index exists at '{CHROMA_PERSIST_DIR}' by running 'build_index.py'.")
    print(f"   - Listening on http://127.0.0.1:{args.port}/mcp/v1/get_model_context")
    try:
        app.run(host='0.0.0.0', port=args.port, debug=False)
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"\n❌ ERROR: Port {args.port} is already in use.")
            print("   You can either stop the other program or run this server on a different port:")
            print(f"   python local_mcp_server.py --port {args.port + 1}")