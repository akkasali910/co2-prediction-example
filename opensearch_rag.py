#!/usr/bin/env python3
"""
A console-based RAG application that uses a local OpenSearch index as its
knowledge base and a local Ollama model for generation.
This script is a refactor of log_analyzer.py to use OpenSearch instead of ChromaDB.
"""

import os
import sys
import logging

from opensearchpy import OpenSearch
# --- LlamaIndex Imports ---
from llama_index.core.llms import ChatMessage
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- Local Imports from original script ---
from config import (OLLAMA_MODEL, OLLAMA_REQUEST_TIMEOUT, EMBEDDING_MODEL_NAME,
                    TOP_K_CHUNKS)

# --- OpenSearch Configuration ---
OPENSEARCH_ENDPOINT = "http://localhost:9200"
OPENSEARCH_INDEX_NAME = "companies_house" # Correct index name for your data
VECTOR_DIMENSION = 384 # Based on 'all-MiniLM-L6-v2'

# --- Disable Telemetry/Warnings ---
os.environ["LLAMA_INDEX_DO_NOT_TRACK"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_services():
    """Initializes the OpenSearch client, embedding model, and LLM."""
    logging.info("\n--- 1. Initializing Ollama LLM ---")
    try:
        llm = Ollama(
            model=OLLAMA_MODEL,
            request_timeout=OLLAMA_REQUEST_TIMEOUT,
            temperature=0.1,
            stop=["\n\n", "---"]
        )
        llm.complete("Hi", max_tokens=1) # Test call
        logging.info(f"✅ Ollama LLM '{OLLAMA_MODEL}' initialized successfully.")
    except Exception as e:
        logging.error(f"❌ Ollama ERROR: Model '{OLLAMA_MODEL}' not accessible or server not running.")
        logging.error(f"   Action: Ensure 'ollama serve' is running and you have pulled the model: 'ollama pull {OLLAMA_MODEL}'")
        sys.exit(1)

    logging.info("\n--- 2. Initializing Embedding Model and OpenSearch Client ---")
    try:
        # Initialize Embeddings
        embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
        logging.info(f"✅ Embedding Model '{EMBEDDING_MODEL_NAME}' initialized.")

        # Initialize a direct OpenSearch client
        opensearch_client = OpenSearch(
            hosts=[OPENSEARCH_ENDPOINT],
            http_compress=True,
            use_ssl=False,
            verify_certs=False,
            ssl_show_warn=False,
        )
        if not opensearch_client.ping():
            raise ConnectionError("Could not connect to OpenSearch.")
        logging.info(f"✅ OpenSearch Client connected to index '{OPENSEARCH_INDEX_NAME}'.")

        return opensearch_client, embed_model, llm
    except Exception as e:
        logging.error(f"❌ Failed to initialize clients. Error: {e}")
        logging.error(f"   Action: Ensure OpenSearch is running and the index has been created and populated. Error: {e}")
        sys.exit(1)

def perform_hybrid_search(client: OpenSearch, embed_model, query_str: str, top_k: int) -> str:
    """Performs a hybrid search directly against OpenSearch and returns formatted context."""
    query_embedding = embed_model.get_query_embedding(query_str)

    hybrid_query = {
        "query": {
            "hybrid": {
                "queries": [
                    {"multi_match": {"query": query_str, "fields": ["combined_text", "company_name^2"]}},
                    {"knn": {"company_vector": {"vector": query_embedding, "k": top_k}}}
                ]
            }
        }
    }

    response = client.search(index=OPENSEARCH_INDEX_NAME, body=hybrid_query, size=top_k)

    # Extract and format context from results
    context_parts = []
    seen_ids = set()
    for hit in response["hits"]["hits"]:
        if hit["_id"] in seen_ids:
            continue
        seen_ids.add(hit["_id"])
        source = hit.get("_source", {})
        details = [f"Company Name: {source.get('company_name', 'N/A')}"]
        if source.get("company_status"): details.append(f"Status: {source.get('company_status')}")
        if source.get("incorporation_date"): details.append(f"Incorporated On: {source.get('incorporation_date')}")
        if source.get("address_line1"): details.append(f"Address: {source.get('address_line1')}")
        if source.get("locality"): details.append(f"City: {source.get('locality')}")
        if source.get("sic_codes"): details.append(f"SIC Codes: {', '.join(source.get('sic_codes', []))}")
        context_parts.append("===\n" + "\n".join(details))

    return "\n".join(context_parts)

def analyze_console():
    """Main console loop for chatting about the local index."""
    opensearch_client, embed_model, llm = setup_services()
    chat_history: list[ChatMessage] = []

    try:
        print("\n=======================================================")
        print("      OpenSearch Companies House Analyzer is READY (Conversational Mode)")
        print("=======================================================")
        print(f"   - Knowledge Base: OpenSearch index '{OPENSEARCH_INDEX_NAME}'")
        print("   - Enter your query or type 'exit' or 'quit' to end. Type 'reset' to clear history.")
        print("=======================================================\n")

        while True:
            user_query = input("❓ Your Query: ")

            if user_query.strip().lower() in ['exit', 'quit']:
                break
            if not user_query.strip():
                continue
            if user_query.strip().lower() == 'reset':
                chat_history = []
                print("Conversation history has been reset.")
                print("-------------------------------------------------------\n")
                continue

            # 1. Retrieve context from OpenSearch
            print("\n🔍 Retrieving context from OpenSearch...")
            context = perform_hybrid_search(opensearch_client, embed_model, user_query, TOP_K_CHUNKS)

            if not context:
                print("   ...No relevant context found in OpenSearch.")
                context = "No context found."
            else:
                print("   ...Context found. Preparing response.")
                print("\n--- RETRIEVED CONTEXT ---")
                print(context.strip())
                print("-------------------------")

            # 2. Build the prompt for the LLM
            system_prompt = (
                "You are an expert on UK Companies House data. Your task is to answer the user's questions based on the provided context. "
                "The context is retrieved from an OpenSearch knowledge base that contains Companies House data. This system originally analyzed local directory files. "
                "Always be helpful and answer concisely. If the answer is not in the context, "
                "state that you cannot find the answer in the provided documents. "
                "You have access to the conversation history."
            )
            messages = [ChatMessage(role="system", content=system_prompt)]
            messages.extend(chat_history)
            messages.append(ChatMessage(role="user", content=f"Context:\n{context}\n\nQuestion: {user_query}"))

            # 3. Stream the response from the LLM
            print("\n🤖 AI Response (Streaming):")
            response_text = ""
            for token in llm.stream_chat(messages):
                response_text += token.delta
                print(token.delta, end="", flush=True)

            chat_history.append(ChatMessage(role="user", content=user_query))
            chat_history.append(ChatMessage(role="assistant", content=response_text))

            print("\n-------------------------------------------------------\n")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        logging.error(f"\n❌ An unexpected error occurred: {e}")
    finally:
        print("\n👋 Exiting analyzer. Goodbye!")

if __name__ == '__main__':
    analyze_console()